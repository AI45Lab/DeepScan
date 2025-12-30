# LLM-Diagnose FastAPI（用户使用指南）

这是一个最小的 REST 风格服务，帮你通过 HTTP 启动诊断任务并查询结果。

## 安装

```bash
cd /root/code/LLM-Diagnose-Framework
pip install -e .[all]          # 安装 fastapi/uvicorn 和框架依赖
# 或：pip install -e . && pip install fastapi uvicorn
```

## 配置

- `DIAGNOSE_CONFIG`（可选）：诊断配置文件路径（YAML/JSON），默认 `examples/config.xboundary.yaml`。
- `DIAGNOSE_DRY_RUN`（可选）：设为 `true` 仅做校验，不实际加载模型/数据。
- `DIAGNOSE_OUTPUT_DIR`（可选）：结果输出目录，默认 `results/`。
- `API_KEY`（可选，默认如下）：除 `/health` 外所有接口都需要 `Authorization: Bearer <token>`。默认值：`iM1b1sxY8yCYCACqA7lvHEdh1XjpKgS4`，可通过环境变量覆盖。

## 启动服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 接口

- `GET /health`：存活探针。
- `POST /evaluations`：创建一个诊断任务，立即返回 `run_id`（状态 202）。请求体可选字段 `job_id`（兼容旧字段 `run_id`），若不提供或为空则自动生成。
- `GET /evaluations/{run_id}`：根据 `run_id` 查询任务状态/结果，若在运行会等待完成。
- `GET /evaluations`：仅返回当前内存中的 `run_id` 列表。
- `GET /runs/{run_id}`：`/evaluations/{run_id}` 的别名。

## 使用示例

1) 发起任务（始终异步返回）：
```bash
curl -X POST http://localhost:8000/evaluations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"job_id": ""}'    # job_id 可省略/留空，服务端会自动生成
# 返回: {"run_id":"<uuid>","status":"pending",...}
```

2) 轮询任务状态/结果：
```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/evaluations/<run_id>
# status: pending|running|completed|failed
```

3) 任务完成后：
- 响应里的 `result` 就是 `run_from_config` 的输出。
- 磁盘上也会有产物：`DIAGNOSE_OUTPUT_DIR/<run_id>/`。

4) 仅做校验（不实际运行）：
```bash
DIAGNOSE_DRY_RUN=true uvicorn api.main:app --reload
```
这会验证配置和注册表，但不会加载模型/数据，返回一个简短结果。

