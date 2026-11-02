"""Submit localization experiments to SLURM as a job array (one task per combination).

    python -m hpc.submit --dataset keble --retriever salad --matcher lightglue --limit 20 \\
        --qos boost_qos_dbg --time 00:25:00
    python -m hpc.submit --dataset hge,cab,lin --all --warmup
    python -m hpc.submit --dataset christchurch --all --resume
    python -m hpc.submit --progress
    python -m hpc.submit --progress --only partial
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loc.datasets import SCENES, Dataset
from loc.matchers import MATCHER_NAMES
from loc.paths import OUTPUT_ROOT, REPO_ROOT, experiment_name
from loc.retrievers import RETRIEVER_NAMES

SBATCH_SCRIPT = REPO_ROOT / "hpc" / "localize.sbatch"
JOBS_DIR = REPO_ROOT / "jobs"
LOGS_DIR = REPO_ROOT / "logs"


def parse_list(value, allowed, label):
    if value is None:
        return None
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in allowed]
    if unknown or not names:
        raise SystemExit(f"invalid --{label} {unknown or value!r}: choose from {list(allowed)}")
    return names


def build_combos(args):
    """Cartesian product of datasets x retrievers x matchers. --all fills the unspecified fields."""
    datasets = parse_list(args.dataset, SCENES, "dataset")
    retrievers = parse_list(args.retriever, RETRIEVER_NAMES, "retriever")
    matchers = parse_list(args.matcher, MATCHER_NAMES, "matcher")
    if args.all:
        retrievers = retrievers or list(RETRIEVER_NAMES)
        matchers = matchers or list(MATCHER_NAMES)
    if not retrievers or not matchers:
        raise SystemExit("specify --retriever and --matcher, or use --all")
    return [(d, r, m, args.k) for d in datasets for r in retrievers for m in matchers]


def localize_options(args, warmup=False):
    """Options forwarded to loc.localize for every task of the array."""
    if warmup:
        return ["--limit", "1"]
    opts = []
    if args.limit:
        opts += ["--limit", str(args.limit)]
    if args.query_stride != 1:
        opts += ["--query-stride", str(args.query_stride)]
    if args.resume:
        opts.append("--resume")
    for o in args.retriever_opt or []:
        opts += ["--retriever-opt", o]
    for o in args.matcher_opt or []:
        opts += ["--matcher-opt", o]
    return opts


def write_plan(combos, tag, options_for):
    """Plan file: one tab-separated line per array task, read by localize.sbatch.

    The name is never reused: tasks read the plan when they start, so overwriting the plan of a
    queued array would make it run the wrong combinations.
    """
    JOBS_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    path, n = JOBS_DIR / f"{stamp}_{tag}.tsv", 1
    while path.exists():
        path, n = JOBS_DIR / f"{stamp}_{tag}_{n}.tsv", n + 1
    # The options column is split on whitespace by localize.sbatch, so values cannot contain spaces.
    lines = ["\t".join([d, r, m, str(k), " ".join(options_for(d, r, m, k))])
             for d, r, m, k in combos]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def sbatch_command(args, plan, n_tasks, name, warmup=False, dependency=None):
    log = LOGS_DIR / (f"{name}_%A_%a.out" if n_tasks > 1 else f"{name}_%j.out")
    cmd = ["sbatch", f"--account={args.account or '<account>'}", f"--job-name={name}",
           f"--qos={args.qos}",
           f"--time={args.warmup_time if warmup else args.time}", f"--output={log}",
           f"--chdir={REPO_ROOT}", f"--export=ALL,LOC_JOB_PLAN={plan}"]
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if warmup:
        cmd.append("--parsable")   # print only the job id, used for --dependency
    if dependency:
        # The array starts only if every warm-up task succeeded.
        cmd.append(f"--dependency=afterok:{dependency}")
    if n_tasks > 1:
        span = f"0-{n_tasks - 1}"
        cmd.append(f"--array={span}%{args.throttle}" if args.throttle else f"--array={span}")
    cmd.append(str(SBATCH_SCRIPT))
    return cmd


def submit(cmd, dry_run):
    print("$ " + " ".join(cmd))
    if dry_run:
        return "<warmup-job-id>"
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout, end="")
    print(out.stderr, end="", file=sys.stderr)
    if out.returncode != 0:
        raise SystemExit(out.returncode)
    return out.stdout.strip().split(";")[0]


def parse_squeue(text):
    """squeue rows (job id, state, elapsed, reason) -> {task index: (state, elapsed, reason)}.

    A task of an array is '123_4' and a single job '123' is task 0. The tasks of an array that are
    still queued share one row, '123_[3-62%4]' or '123_[5,7]', which is expanded here.
    """
    states = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 4:
            continue
        job, info = parts[0], tuple(parts[1:4])
        queued = re.match(r"^\d+_\[(.+)\]$", job)
        if queued:
            for chunk in queued.group(1).split("%")[0].split(","):
                lo, _, hi = chunk.partition("-")
                for i in range(int(lo), int(hi or lo) + 1):
                    states[i] = info
        else:
            states[int(job.partition("_")[2] or 0)] = info
    return states


def squeue_states(job_name):
    try:
        out = subprocess.run(["squeue", "--me", "-h", f"--name={job_name}", "-o", "%i\t%T\t%M\t%R"],
                             capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return parse_squeue(out.stdout) if out.returncode == 0 else None


def latest_logs(job_name):
    """{task index: most recent log file}. Warm-up logs have another job name and do not match."""
    pattern = re.compile(rf"^{re.escape(job_name)}_\d+(?:_(\d+))?\.out$")
    logs = {}
    for path in sorted(LOGS_DIR.glob("*.out"), key=lambda p: p.stat().st_mtime):
        m = pattern.match(path.name)
        if m:
            logs[int(m.group(1) or 0)] = path
    return logs


def last_line(path):
    if path is None:
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [line.rstrip() for line in lines if line.strip()]
    return lines[-1][:48] if lines else ""


def expected_queries(dataset, limit, stride, cache):
    """Number of queries of a task, None if the data of the scene is not on this machine."""
    if (dataset, stride) not in cache:
        try:
            cache[dataset, stride] = len(Dataset(dataset, query_stride=stride).queries())
        except OSError:
            cache[dataset, stride] = None
    total = cache[dataset, stride]
    return min(total, limit) if total and limit else total


def plan_rows(plan):
    """Current state of every combination of a plan.

    squeue knows what is running or queued but forgets finished jobs, the result files measure the
    work done and survive the job, and the last log line tells how a task is doing or how it died.
    """
    # Plan files are named <date>_<time>_<job name>.tsv.
    job_name = plan.stem[len("YYYYmmdd_HHMMSS_"):]
    states, logs = squeue_states(job_name), latest_logs(job_name)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--name", default=None)

    rows, cache = [], {}
    for i, line in enumerate(plan.read_text(encoding="utf-8").splitlines()):
        dataset, retriever, matcher, k, *rest = line.split("\t")
        opts = parser.parse_known_args(rest[0].split() if rest else [])[0]
        name = opts.name or experiment_name(dataset, retriever, matcher, k)
        done = sum(1 for _ in (OUTPUT_ROOT / "localization" / name / "results").glob("*_result.json"))
        total = expected_queries(dataset, opts.limit, opts.query_stride, cache)

        state, elapsed, reason = (states or {}).get(i, ("", "", ""))
        if state == "PENDING" and reason and reason != "None":
            state = f"PEND:{reason[:5]}"
        elif not state:
            # Not in squeue: the results tell a finished task from one to run again with --resume.
            if total and done >= total:
                state = "done"
            elif done:
                state = "partial"
            else:
                state = "ended" if i in logs else "-"
        rows.append({"i": i, "dataset": dataset, "name": name, "state": state, "elapsed": elapsed,
                     "done": done, "total": total, "last": last_line(logs.get(i))})
    return rows, states is not None


def row_matches(row, only):
    """--only: a state (partial, running, done, pending, idle, ended, todo) or part of a name."""
    key = only.lower()
    if key in ("pending", "pend"):
        return row["state"].startswith("PEND")
    if key in ("running", "run"):
        return row["state"] == "RUNNING"
    if key in ("partial", "part"):
        return row["state"] == "partial"
    if key in ("done", "ended"):
        return row["state"] == key
    if key == "idle":
        return row["state"] == "-"
    if key == "todo":
        return row["state"] != "done"
    return key in row["name"].lower()


def show_progress(plan, only=None, detail=False):
    if plan is None:
        plans = sorted(JOBS_DIR.glob("*.tsv"))
        plans = [p for p in plans if "_warmup" not in p.name] or plans
        if not plans:
            raise SystemExit(f"no plan in {JOBS_DIR}: submit an array first, or use --plan")
        plan = plans[-1]
    rows, squeue_ok = plan_rows(plan)

    # The summary per scene comes first: it answers "how far are we" without scrolling through
    # every combination, and it counts all of them even when --only hides some.
    print(f"plan: {plan.name} ({len(rows)} combinations)\n")
    print(f"{'scene':<14} {'comb.':>6} {'done':>5} {'run':>4} {'pend':>5} {'part':>5} {'idle':>5}"
          f"   {'queries':>16}   progress")
    print("-" * 100)
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        states = [r["state"] for r in group]
        pending = sum(s.startswith("PEND") for s in states)
        idle = states.count("-") + states.count("ended")
        q_done, q_total = sum(r["done"] for r in group), sum(r["total"] or 0 for r in group)
        pct = min(100 * q_done / q_total, 100) if q_total else 0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"{dataset:<14} {len(group):>6} {states.count('done'):>5} {states.count('RUNNING'):>4} "
              f"{pending:>5} {states.count('partial'):>5} {idle:>5}"
              f"   {q_done:>7}/{q_total:<8} {bar} {pct:3.0f}%")

    q_done, q_total = sum(r["done"] for r in rows), sum(r["total"] or 0 for r in rows)
    n_partial = sum(1 for r in rows if r["state"] == "partial")
    print("-" * 100)
    print(f"total: {sum(1 for r in rows if r['state'] == 'done')}/{len(rows)} combinations complete, "
          f"{sum(1 for r in rows if r['state'] == 'RUNNING')} running, {q_done}/{q_total} queries "
          f"({100 * q_done / q_total if q_total else 0:.0f}%)")
    if n_partial:
        print(f"  {n_partial} interrupted: list them with --progress --only partial, then submit "
              "them again with --resume")
    if not squeue_ok:
        print("  squeue is not available: states come only from the results and the logs")

    if not (detail or only):
        print("\n(--detail shows every combination, --only <state|name> filters them)")
        return 0
    shown = [r for r in rows if row_matches(r, only)] if only else rows
    print(f"\n{len(shown)}/{len(rows)} combinations" + (f" [--only {only}]" if only else ""))
    if not shown:
        return 0
    print(f"{'#':>3} {'combination':<42} {'SLURM':<11} {'time':>9} {'queries':>13} {'%':>5}  last log line")
    print("-" * 124)
    for r in shown:
        queries = f"{r['done']:>6}/{r['total'] or '?':<6}"
        pct = f"{100 * r['done'] / r['total']:4.0f}%" if r["total"] else "    -"
        print(f"{r['i']:>3} {r['name']:<42} {r['state']:<11} {r['elapsed']:>9} {queries} {pct}  {r['last']}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help=f"comma-separated, from {list(SCENES)}")
    p.add_argument("--retriever", help=f"comma-separated, from {list(RETRIEVER_NAMES)}")
    p.add_argument("--matcher", help=f"comma-separated, from {list(MATCHER_NAMES)}")
    p.add_argument("--all", action="store_true",
                   help="use every registered retriever/matcher for the fields not given")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--query-stride", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="skip queries that already have a result")
    p.add_argument("--retriever-opt", action="append", metavar="KEY=VALUE")
    p.add_argument("--matcher-opt", action="append", metavar="KEY=VALUE")
    p.add_argument("--warmup", action="store_true",
                   help="first build the retrieval cache once per (dataset, retriever), then run "
                        "the array after it")
    p.add_argument("--warmup-time", default="01:00:00")
    p.add_argument("--account", default=os.environ.get("LOC_SLURM_ACCOUNT"),
                   help="SLURM account (default: $LOC_SLURM_ACCOUNT)")
    p.add_argument("--partition", default=None, help="default: the one in localize.sbatch")
    p.add_argument("--qos", default="normal")
    p.add_argument("--time", default="06:00:00", help="walltime of each task")
    p.add_argument("--throttle", type=int, default=0, help="maximum number of tasks running at once")
    p.add_argument("--job-name", default=None)
    p.add_argument("--dry-run", action="store_true", help="write the plan and print the commands only")
    p.add_argument("--status", action="store_true", help="show your queued and running jobs")
    p.add_argument("--progress", action="store_true",
                   help="state of the last submitted plan: SLURM state, queries done and last log "
                        "line of every combination")
    p.add_argument("--plan", type=Path, default=None,
                   help="with --progress: a plan file instead of the last one in jobs/")
    p.add_argument("--only", default=None,
                   help="with --progress: show only the combinations in a state (partial, running, "
                        "pending, done, ended, idle, todo) or with this text in their name")
    p.add_argument("--detail", action="store_true",
                   help="with --progress: show every combination")
    return p.parse_args()


def main():
    args = parse_args()
    if args.status:
        return subprocess.run(["squeue", "--me", "-o", "%.10i %.24j %.8T %.10M %.10l %R"]).returncode
    if args.progress:
        return show_progress(args.plan, args.only, args.detail)
    if not args.dataset:
        raise SystemExit("--dataset is required")
    if not args.account and not args.dry_run:
        raise SystemExit("missing SLURM account: use --account or export LOC_SLURM_ACCOUNT")

    combos = build_combos(args)
    tag = args.job_name or "loc_" + "-".join(sorted({c[0] for c in combos}))
    LOGS_DIR.mkdir(exist_ok=True)

    dependency = None
    if args.warmup:
        # The retrieval cache depends only on (dataset, retriever): one task per pair builds it,
        # so that the matchers of the main array do not all rebuild it at the same time.
        pairs = {}
        for d, r, m, k in combos:
            pairs.setdefault((d, r), (d, r, m, k))
        warmup = list(pairs.values())
        plan = write_plan(warmup, f"{tag}_warmup", lambda d, r, m, k: [
            *localize_options(args, warmup=True), "--name", f"_warmup_{d}_{r}"])
        print(f"warm-up: {len(warmup)} caches to build, one per (dataset, retriever) -> {plan}")
        for i, (d, r, m, k) in enumerate(warmup):
            print(f"  [{i}] {d} {r} (the matcher {m} does not matter here)")
        dependency = submit(sbatch_command(args, plan, len(warmup), f"{tag}_warmup", warmup=True),
                            args.dry_run)
        if not args.dry_run:
            print(f"warm-up submitted: job {dependency}")

    plan = write_plan(combos, tag, lambda d, r, m, k: localize_options(args))
    print(f"{len(combos)} tasks -> {plan}")
    for i, (d, r, m, k) in enumerate(combos):
        print(f"  [{i}] {experiment_name(d, r, m, k)}")
    submit(sbatch_command(args, plan, len(combos), tag, dependency=dependency), args.dry_run)
    if args.dry_run:
        print("(--dry-run: nothing was submitted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
