# Qoder2API (Hermes 专属部署配置指南)

本项目已成功部署至 `H:\qoder2api`，并完成了独立的 Python 虚拟环境与依赖构建。

---

## 1. 快速使用

### 启动服务
- **前台运行（查看实时日志）**：双击运行 `start.cmd`
- **后台静默运行（无黑框，推荐）**：双击运行 `start-silent.vbs`（日志自动保存在 `qoder2api.log`）

### 检查与停止
- **状态检查**：双击运行 `status.cmd`，自动检测 8963 端口并拉取模型列表
- **安全停止**：双击运行 `stop.cmd`，精准停止占用 8963 端口的进程（不会误杀系统其他 Python 进程）

---

## 2. 获取 Qoder 访问令牌 (PAT)

1. 打开浏览器登录 [Qoder 账号集成中心](https://qoder.cn/account/integrations)
2. 在「**服务集成**」中点击「**创建个人访问令牌**」（Personal Access Token）
3. 复制生成的令牌（形如 `pt-xxxx_019exxxx-...`）
4. 该令牌作为 API Key 即可鉴权使用。

---

## 3. 接入 Hermes 配置范例

若需要将 Qoder 提供的模型接入 Hermes，可在 Hermes 的 `config.yaml` 中增加 `custom_providers`：

```yaml
custom_providers:
  qoder:
    base_url: "http://127.0.0.1:8963/v1"
    api_key: "pt-xxxx_你的Qoder_PAT令牌"
    models:
      - Qwen3.8-Max-Preview
      - Qwen3.7-Max
      - Qwen3.7-Plus
      - DeepSeek-V4-Pro
```

---

## 4. 技术特性与 Windows 优化

- **协议兼容**：全面兼容 OpenAI `/v1/chat/completions` 与 `/v1/models`，支持流式 SSE、工具调用与推理。
- **环境隔离**：虚拟环境独立建立在 `H:\qoder2api\.venv`，完全不占用系统 C 盘。
- **Windows 代理避坑**：启动脚本内已内置 `set NO_PROXY=*`，彻底防止 Windows 下 httpx 读取 IE 系统代理导致的请求超时。

---

## 5. 每日 100 Credits 自动领取（签到）

服务启动后自动运行（随反代进程，无需单独脚本）。只要"当天还没领取"，无论几点启动、是否错过 10:00 刷新，都会自动补领；领取成功后当天不再重复，等到下一个刷新窗口（活动每日 10:00 UTC+8 刷新，窗口取 10:15，留 15 分钟余量）。活动一直未开放的话，最多查到 12:00 就当天放弃、次日窗口再看——不会整天轮询。

配置（任选一种，环境变量优先）：

1. 项目目录下 `checkin.json`（推荐；已加入 .gitignore，不会进版本库）：

```json
{
  "pat": "pt-你的Qoder_PAT",
  "retry_minutes": 30
}
```

多账号：`"pats": ["pt-一", "pt-二"]`

2. 环境变量：`QODER_CHECKIN_PAT`（多个用英文逗号分隔）、`QODER_CHECKIN_RETRY_MINUTES`（未领取时的重试间隔，默认 30 分钟）

说明：

- 认证复用桥自身会话（PAT → jobToken），无需额外配置
- 领取过程与结果打印在服务日志里（`[checkin]` 前缀），成功示例如：`[checkin] nickXXXX: 签到成功 +100 积分`
- 不配置则功能自动关闭，不影响主链路

---

## 6. 指标透传（缓存命中率 / 每秒 token 数）

上游在每次回答末尾会下发 usage（prompt / completion / total / `prompt_tokens_details.cached_tokens` / `credits`）。本桥原先把 usage 写死成 0，导致 Hermes 侧拿不到缓存命中率和每秒输出 token 数；现已改为真实透传：

- **流式**：在 `[DONE]` 之前补一个 OpenAI `stream_options.include_usage` 风格的 usage chunk（`choices: []`）
- **非流式**：响应体的 `usage` 直接来自上游
- `prompt_tokens_details.cached_tokens` 是客户端算缓存命中率的标准字段；`completion_tokens` 是算每秒 token 数的分子
- 上游没上报的字段不写入响应（避免把"没上报"伪装成 0）
- 额外带上上游的 `credits`（本次消耗积分），便于核对用量

实现：`usage.py`（提取 + 映射，纯函数、可单测）+ `transform.extract_usage_line`（信封解包）。

---

## 7. 上游"忙 / 排队"的处理

上游繁忙时会用 `code 10605` 之类的信封（常带 403）告知"忙 / 排队"。这类回应**不是鉴权失败**：

- 不再误判为 401 → 不再白刷新会话（旧行为会轮换 token，连累其他在途请求）
- 信封可能嵌套多层（实测 2 层：外层 `code 403` → 内层 `code 10605` → 队列详情）；解析会逐层下钻到含 `retryAfterSeconds` 的详情层，`Retry-After` 才不会漏
- 非流式：返回 **HTTP 503 + `Retry-After`**（上游给了 `retryAfterSeconds` 时），`error.type = "upstream_busy"`
- 流式：流内错误 chunk 的 `error.type = "upstream_busy"`，客户端可据此退避重试
- 错误消息里**故意不写** `(retry after Ns)` 字样：桌面端(Hermes)会把消息里解析出的任何重试时间渲染成"限额将于 X 重置"倒计时（把排队误当额度重置）；重试建议只走 `Retry-After` 头与消息 JSON 里的 `retryAfterSeconds`
- 真正的登录过期（`code 105` / 401 / 403 无忙标记）仍照旧刷新会话重试一次

实现：`qoder_auth.QoderBusyError` + `detect_upstream_busy()`（分类 + 多层信封下钻），桥接层负责状态码映射。

---

## 8. 多账号池（一个服务管理多个 Qoder 账号）

参考 workbuddy2api 的账号池模式。客户端**双轨鉴权**：

- Bearer 以 `pt-` 开头 → 直通该 PAT 账号（旧行为，Hermes 现有配置零改动）
- Bearer 为配置的 `gateway_key` → 服务端从账号池选号，客户端只需一把网关钥匙

### 配置（pool.json，模板见 pool.json.example）

```json
{
  "gateway_key": "随便一长串随机字符串",
  "pats": ["pt-账号一", "pt-账号二"],
  "options": {
    "sticky_ttl_minutes": 30,
    "auth_fail_threshold": 2,
    "cooldown_minutes": 15,
    "cooldown_max_minutes": 360,
    "quota_cooldown_minutes": 360
  }
}
```

环境变量优先：`QODER_GATEWAY_KEY`、`QODER_POOL_PATS`（逗号分隔）。
无 pool.json 时回退读 checkin.json 的 pat/pats（旧部署兼容）；两处都没有 = 纯直通模式。

### 选号与故障治理

- 会话粘性：同一对话（conversation_id / prompt_cache_key / 首条 user 消息哈希）固定同一账号，30 分钟滚动续期，保证多轮上下文与上游 prompt cache 不跳号
- 无粘性键时按最近最少使用（LRU）均摊流量
- 同步阶段（冷交换/建会话）失败 → 自动换号重试，单请求上限 3 次
- 连续鉴权失败达到阈值 → 指数退避冷却（15 分钟起、翻倍、封顶 6 小时），成功一次清零
- 每日 30 分钟周期巡检 `user_status`：`isQuotaExceeded` 的号硬冷却到 `nextResetAt`，恢复后自动回池
- 上游忙/排队（10605）与模型不支持等客户端错误**不计入账号过错**（换号无意义，原样上抛）
- 全池冷却时仍临时放行最早恢复的账号，不拒绝服务

### 观测

`GET /status` 返回各账号脱敏状态（昵称/尾号、ok|cooldown|quota_exceeded、冷却剩余、连续失败数、quota、nextResetAt），不含 PAT 明文。

### 热加载（加账号不用重启）

服务每次请求前检查 `pool.json` / `checkin.json` 的文件指纹（2 秒节流），变化即增量合并：新账号即时进池，已有账号的冷却/失败计数/昵称全部保留，被删账号自动作废其会话粘性。签到任务每轮开始前也重读名单。即：**双击 add-pat.cmd 加号 → 下一个请求/签到周期自动生效**。配置目录可用 `QODER_PROJECT_DIR` 外置（多实例场景）。

### 一键加号：add-pat.cmd

对标 workbuddy2api 的 login-helper 交互。双击运行（或在 cmd 里带参数 `add-pat.cmd pt-xxx ...`）：

1. 命令窗口粘贴 PAT（输入不回显，不落日志）
2. 立即向 Qoder 网关验证（jobToken 冷交换确认真实昵称，坏号/格式错当场拒绝，绝不写入）
3. 合并写入 `pool.json`；首次使用自动生成随机 `gateway_key` 并在窗口打印
4. 若只有 `checkin.json` 旧配置，其中 PAT 自动并入 pool.json（一次性迁移）
5. 检测到服务在跑则提示热加载自动生效，无需重启

### 签到与池共用账号

每日 100 Credits 签到名单 = `checkin.json` 的 pat/pats **∪ 账号池全部 PAT**（pool.json / QODER_POOL_PATS，自动去重并入）——往池里加新账号，签到不用另配，配合热加载下一个签到周期即吃。仅当显式设置 `QODER_CHECKIN_PAT` 时才完全覆盖、不并池。

---

## 9. 思考强度档位透传（reasoning_effort）

客户端（如 Hermes custom provider）在 OpenAI 请求**顶层**发 `reasoning_effort`，本桥归一后注入 Qoder 上游 `parameters:{enable_thinking:true, reasoning_effort:low|medium|xhigh}`（上游实测只认三档，无 high）：

| 客户端档位 | 注入上游 |
|---|---|
| none / minimal / low | low |
| medium / high | medium |
| xhigh / max / ultra | xhigh |
| 未传 / 未知值 | 不注入（上游默认 medium，与旧行为一致） |

Hermes 侧 `agent.reasoning_effort` 经 custom profile 钳到 OpenAI 兼容集后发出（如 ultra → 线上为 max → 本桥归一 xhigh）。

真机实测（2026-09-22，Qwen3.8-Flash 同题推理）：low 档 reasoning_tokens 4096（127s）↔ xhigh 档 11590（245s），约 2.8 倍，档位真实生效。

> `none` 暂映射到 low 而非关闭思考——上游是否接受 `enable_thinking:false` 未实测，不冒进。
