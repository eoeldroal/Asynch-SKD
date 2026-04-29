from WebOSWorld.mock_server.create_mock_web_osgym_dataset import build_rows


def test_build_rows_defaults_to_web_skd_agent_for_existing_smoke_runs():
    rows = build_rows(split="train", num_samples=1, task_id_start=12345)

    assert rows[0]["agent_name"] == "web_skd_agent"
    assert rows[0]["extra_info"]["tools_kwargs"]["computer"]["create_kwargs"]["task_id"] == "12345"


def test_build_rows_can_target_fully_async_web_tool_agent():
    rows = build_rows(split="train", num_samples=2, task_id_start=12345, agent_name="web_tool_agent")

    assert [row["agent_name"] for row in rows] == ["web_tool_agent", "web_tool_agent"]
    assert [row["extra_info"]["task_id"] for row in rows] == ["12345", "12346"]
