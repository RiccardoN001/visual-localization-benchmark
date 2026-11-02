from hpc import submit
from loc.capture_io import save_pinhole_sensors


def write_scene(data_root, n_queries):
    capture = data_root / "KEBLE" / "capture"
    capture.mkdir(parents=True)
    save_pinhole_sensors([("cam0", "test", 640, 480, 500, 500, 320, 240)], capture / "sensors.txt")
    (capture / "trajectories.txt").write_text("# none\n", encoding="utf-8")
    rows = "".join(f"{i}, cam0, images/{i}.jpg\n" for i in range(n_queries))
    (capture / "images.txt").write_text(rows, encoding="utf-8")


def write_results(output_root, name, n):
    results = output_root / "localization" / name / "results"
    results.mkdir(parents=True)
    for i in range(n):
        (results / f"{i}_result.json").write_text("{}", encoding="utf-8")


def setup_workspace(tmp_path, monkeypatch, states):
    for name in ("jobs", "logs", "output"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(submit, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(submit, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(submit, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(submit, "squeue_states", lambda job_name: states)
    data_root = tmp_path / "data"
    original = submit.Dataset
    monkeypatch.setattr(submit, "Dataset",
                        lambda name, query_stride=1: original(name, data_root, query_stride))
    write_scene(data_root, 10)
    return tmp_path


def write_plan(root, lines, tag="loc_keble"):
    plan = root / "jobs" / f"20260101_120000_{tag}.tsv"
    plan.write_text("".join("\t".join(line) + "\n" for line in lines), encoding="utf-8")
    return plan


def combo(matcher, options=""):
    return ["keble", "salad", matcher, "3", options]


def test_squeue_rows_are_mapped_to_task_indices():
    states = submit.parse_squeue("123_4\tRUNNING\t1:00\tnode1\n"
                                 "123_[6-7,9%2]\tPENDING\t0:00\tPriority\n"
                                 "456\tRUNNING\t2:00\tnode2\n")
    assert states[4] == ("RUNNING", "1:00", "node1")
    assert sorted(states) == [0, 4, 6, 7, 9]
    assert states[9] == ("PENDING", "0:00", "Priority")
    assert states[0][1] == "2:00"


def test_state_of_each_combination(tmp_path, monkeypatch):
    root = setup_workspace(tmp_path, monkeypatch, {3: ("RUNNING", "5:00", "node1"),
                                                   4: ("PENDING", "0:00", "Priority")})
    plan = write_plan(root, [combo("lightglue"), combo("superglue"), combo("roma_turbo"),
                             combo("omniglue"), combo("dinov2_matcher"), combo("salad_matcher")])
    write_results(root / "output", "keble_salad_lightglue_k3", 10)
    write_results(root / "output", "keble_salad_superglue_k3", 4)
    (root / "logs" / "loc_keble_77_2.out").write_text("crashed\n", encoding="utf-8")

    rows, squeue_ok = submit.plan_rows(plan)
    assert squeue_ok
    assert [r["state"] for r in rows] == ["done", "partial", "ended", "RUNNING", "PEND:Prior", "-"]
    assert rows[2]["last"] == "crashed"
    assert rows[0]["total"] == 10


def test_limit_and_run_name_are_read_from_the_plan(tmp_path, monkeypatch):
    root = setup_workspace(tmp_path, monkeypatch, None)
    plan = write_plan(root, [combo("lightglue", "--limit 3 --name custom")])
    write_results(root / "output", "custom", 3)

    rows, squeue_ok = submit.plan_rows(plan)
    assert not squeue_ok
    assert (rows[0]["name"], rows[0]["total"], rows[0]["state"]) == ("custom", 3, "done")


def test_logs_of_the_warmup_do_not_count_for_the_main_array(tmp_path, monkeypatch):
    root = setup_workspace(tmp_path, monkeypatch, {})
    (root / "logs" / "loc_keble_warmup_50_0.out").write_text("warm-up\n", encoding="utf-8")
    assert submit.latest_logs("loc_keble") == {}


def test_scene_that_is_not_on_this_machine_has_no_total(tmp_path, monkeypatch):
    root = setup_workspace(tmp_path, monkeypatch, {})
    plan = write_plan(root, [["hge", "salad", "lightglue", "3", ""]], tag="loc_hge")
    rows, _ = submit.plan_rows(plan)
    assert rows[0]["total"] is None
    assert rows[0]["state"] == "-"


def test_progress_table_uses_the_last_plan_and_filters(tmp_path, monkeypatch, capsys):
    root = setup_workspace(tmp_path, monkeypatch, {})
    write_plan(root, [combo("lightglue"), combo("superglue")])
    (root / "jobs" / "20260101_110000_loc_keble_warmup.tsv").write_text("x\n", encoding="utf-8")
    write_results(root / "output", "keble_salad_superglue_k3", 4)

    assert submit.show_progress(None, only="partial") == 0
    out = capsys.readouterr().out
    assert "plan: 20260101_120000_loc_keble.tsv (2 combinations)" in out
    assert "total: 0/2 combinations complete, 0 running, 4/20 queries (20%)" in out
    assert "1/2 combinations [--only partial]" in out
    assert "keble_salad_superglue_k3" in out
    assert "keble_salad_lightglue_k3" not in out
