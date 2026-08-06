"""
Minimal UI i18n for Streamlit (zh / en).

Default language is Chinese (zh). Choice lives in session_state["ui_lang"]
and is mirrored to the ``lang`` query param for shareable / reload persistence.
"""

from __future__ import annotations

from typing import Any, Mapping

LANG_ZH = "zh"
LANG_EN = "en"
DEFAULT_LANG = LANG_ZH
SUPPORTED_LANGS = (LANG_ZH, LANG_EN)

# session_state / query key
LANG_STATE_KEY = "ui_lang"
LANG_QUERY_KEY = "lang"

# Placeholders used by unset selectboxes (both must count as unset).
PLACEHOLDER_ZH = "请选择…"
PLACEHOLDER_EN = "Please select…"
CHOICE_PLACEHOLDERS = frozenset({PLACEHOLDER_ZH, PLACEHOLDER_EN, "请选择...", "Please select..."})

LANG_LABELS = {
    LANG_ZH: "中文",
    LANG_EN: "English",
}

# Stable field keys → display labels (used by missing_start_choices).
START_FIELD_LABELS: dict[str, dict[str, str]] = {
    LANG_ZH: {
        "pair": "货币对",
        "bullish_currency": "看涨货币",
        "peak_engine": "峰值引擎",
        "use_calibrated": "是否使用校准参数",
        "human_review": "不确定证据是否人工确认",
        "bucket_mode": "分档边界方式",
        "pair_mode": "货币对方式",
    },
    LANG_EN: {
        "pair": "Currency pair",
        "bullish_currency": "Bullish currency",
        "peak_engine": "Peak engine",
        "use_calibrated": "Use calibrated params",
        "human_review": "Human review for uncertain evidence",
        "bucket_mode": "Bucket boundary mode",
        "pair_mode": "Pair selection mode",
    },
}

_STRINGS: dict[str, dict[str, str]] = {
    LANG_ZH: {
        # Language selector
        "lang.label": "界面语言",
        "lang.help": "切换后立即生效；会写入网址 ?lang= 参数，刷新仍保留。",
        "report.lang": "报告语言",
        "report.lang.help": "PDF / HTML / Markdown 正文语言。默认「中英双语一起出」：新闻与蒙特卡洛只跑一次，模板各渲染一份。与界面语言独立；源文引用保持原文。",
        "report.lang.both": "中英双语一起出",
        "report.stance_llm": "历史回放也用 LLM 写引用总结",
        "report.stance_llm.help": "默认历史 cheap 路径仅抽取式总结（≈$0）；勾选后按批调用 DeepSeek（约 20 条/次）。",
        "side.cost": "每份报告费用粗估",
        # Password gate
        "auth.caption": "请输入访问密码（可用环境变量 APP_PASSWORD / FX_REPORT_PASSWORD 覆盖默认）。",
        "auth.password": "访问密码",
        "auth.enter": "进入",
        "auth.wrong": "密码错误",
        "auth.empty": "请输入密码",
        # Start setup dialog
        "dlg.start_title": "开始设置",
        "dlg.missing_title": "还不能运行",
        "dlg.setup_mode": "设置模式",
        "dlg.setup_mode.help": "简洁：选定货币对与看涨后，系统自动推荐峰值引擎 / 跳跃 / 方差缩减 / 聚类 / 校准 / 人工确认；专家：以上项须手选，不会静默覆盖。",
        "dlg.simple.caption": "简洁模式：只需货币对 + 看涨货币（可上传单子 PDF/图片）。算法由系统按校准 JSON → 引擎对比 → 产品默认推荐；确认后可在侧栏点「改用专家设置」。",
        "dlg.expert.caption": "专家模式：请逐项选择峰值引擎 / 校准 / 人工确认；不预选默认项。也可上传单子 PDF/图片自动填入货币对与看涨；算法项仍须手选。",
        "dlg.upload_pdf": "上传单子 PDF/图片（可选）",
        "dlg.upload_pdf.label": "上传单子 PDF/图片",
        "dlg.upload_pdf.help": "支持 PDF、JPEG/JPG、PNG。识别货币对、看涨方向、Barrier/Strike/分档等；识别失败可继续手动填，不会编造字段。",
        "dlg.pdf.parsing": "正在解析单子…",
        "dlg.pdf.fail": "单子解析失败，请手动填写。",
        "dlg.pdf.ok": "单子已解析（请核对下方预填项，未识别项仍须手选）。",
        "dlg.pair_mode": "货币对方式（必选）",
        "dlg.pair_mode.hint": "请先选择：目录 或 自定义。",
        "dlg.pair": "货币对（必选）",
        "dlg.pair_custom": "BASE/QUOTE（必选，如 EUR/USD）",
        "dlg.pair_custom.ph": "请输入货币对，例如 EUR/USD",
        "dlg.ticker": "内部符号（可空，默认去掉斜杠）",
        "dlg.bullish": "看涨货币（必选）",
        "dlg.bullish.help": "看涨币走强 = 分析报价升高。选 quote 时自动翻转分析口径。",
        "dlg.bullish.wait": "选定货币对后，再选看涨货币。",
        "dlg.algo.preview": "本次算法由系统推荐",
        "dlg.algo.wait": "选定货币对后，将显示系统推荐的算法组合。",
        "dlg.to_expert": "改用专家设置",
        "dlg.peak_engine": "峰值引擎 peak_engine（必选）",
        "dlg.peak_engine.help": "path_max=离散GBM+Merton跳跃路径最大值；brownian_bridge=日端点间反射原理连续最大值（不含跳跃）",
        "dlg.calibrated": "是否使用校准参数 Stage-1（必选）",
        "dlg.calibrated.help": "使用：优先 output/ 再内置 JSON；不使用：默认先验。不因文件存在而自动勾选。",
        "dlg.human_review": "不确定证据是否人工确认（必选）",
        "dlg.human_review.help": "需要人工确认=低置信度证据先暂停；自动跳过=不打断流水线。",
        "dlg.confirm": "确认开始设置",
        "dlg.cancel": "取消",
        "dlg.algo_fail": "无法生成算法推荐，请先选定货币对。",
        "dlg.missing.caption": "请补全后再点「运行分析」。分档边界方式在主区「概率区间」选择。",
        "dlg.open_start": "打开开始设置",
        "dlg.got_it": "知道了",
        "opt.simple": "简洁（推荐）",
        "opt.expert": "专家",
        "opt.catalog": "目录",
        "opt.custom": "自定义",
        "opt.use": "使用",
        "opt.not_use": "不使用",
        "opt.need_review": "需要人工确认",
        "opt.auto_skip": "自动跳过",
        "opt.cal_use": "使用",
        "opt.cal_skip": "不使用",
        "missing.prefix": "你还没有选择：",
        "missing.empty": "你还没有选择：必选项",
        "placeholder.choice": PLACEHOLDER_ZH,
        # Sidebar
        "side.lang": "语言 / Language",
        "side.start": "① 开始设置",
        "side.start.empty": "尚未完成开始设置（货币对 / 看涨货币等）。",
        "side.start.summary": "**{pair}** · 看涨 **{bull}**  \n模式 `{mode}` · 峰值引擎 `{eng}` · 校准 {cal} · 不确定证据 {hr}",
        "side.start.algo_caption": "本次算法由系统推荐（见结果页审计）。",
        "side.to_expert": "改用专家设置",
        "side.to_expert.help": "打开开始设置并切换到专家模式，可手选峰值引擎 / 校准 / HITL。",
        "side.open_start": "打开开始设置…",
        "side.start.invalid": "开始设置无效：{exc}",
        "side.toc": "侧栏目录：①开始设置 → ②抓取 → ③蒙特卡洛 → ④映射 → ⑤情景 → ⑥证据 → ⑦规则 → ⑧数据源｜分档切点在主区",
        "side.fetch": "② 抓取与判定",
        "side.mc": "③ 蒙特卡洛",
        "side.map": "④ 证据 → 参数映射",
        "side.scenario": "⑤ 情景先验",
        "side.evidence": "⑥ 模板证据计分卡",
        "side.rubric": "⑦ 强弱判定规则",
        "side.sources": "⑧ 数据源状态",
        "side.label_audit": "⑨ 证据人工标注",
        "side.setup_gap": "配置缺口（新闻 Key）",
        "side.setup_gap.body": (
            "未检测到 `NEWSAPI_KEY` / `FINNHUB_API_KEY`。"
            "无 Key 时仍会试央行 RSS + Google News；References 往往偏少。"
            "只填 DeepSeek **不会**自动多出链接。详见 `docs/deploy-docker.md`。"
        ),
        "side.label_audit.has_run": "标注区在主区 **「本次分析审计」正下方**（完整报告与流水线明细之上，不必滚到页底）。",
        "side.label_audit.n": "当前证据条数：{n}",
        "side.label_audit.demo_hint": " · 无证据时可点「加载练习样例」",
        "side.label_audit.jump": "[↓ 跳到证据人工标注](#label-audit-section)",
        "side.label_audit.need_run": "先点主区「运行分析」。标注区会出现在审计面板正下方；即使没有新闻证据，也会显示「怎么填？」与「加载练习样例」。",
        "side.label_audit.learn_progress": "可选进阶：实盘标注 {n}/{need}（够后可勾「使用标签学习到的强度」）",
        "side.label_audit.learn_ready": "标签学习可用（已标 {n} 条）",
        "side.cal.use": "校准参数：开始设置已选「使用」",
        "side.cal.skip": "校准参数：开始设置已选「不使用」",
        "side.cal.unset": "校准参数：尚未在开始设置中选择",
        # Main run / missing
        "main.need_start": "请先完成「开始设置」（侧栏 ①）：货币对、看涨货币、峰值引擎、是否使用校准参数、不确定证据是否人工确认。",
        "main.need_start_short": "请先在侧栏 ① 打开「开始设置」，选好货币对与看涨货币。选好后立刻显示现价与分档设置。",
        "main.run": "运行分析",
        "main.run.need_spot": "需先成功获取现价",
        "main.compare": "双引擎对比",
        "main.compare.help": "path_max vs brownian_bridge（降采样次数，仅 MC）",
        "main.no_spot": "现价未就绪时不能运行分析（分档与 MC 都依赖分析报价）。",
        "main.ready_hint": "确认「开始设置」与概率区间后点「运行分析」。侧栏 ②–⑧ 可调抓取与模型参数；API 可全空。",
        "main.rerun": "再跑一份",
        "main.rerun.help": "用当前开始设置与分档再跑一轮；上一份 PDF/HTML 下载先保留，新结果出来后会替换。",
        "main.new_report": "新报告",
        "main.new_report.help": "打开开始设置改货币对/看涨等；上一份报告下载仍可保留。",
        "main.after_report_hint": "上一份报告已生成。可点「再跑一份」或「新报告」，也可直接再点「运行分析」。",
        "main.window_start": "窗口起点",
        "main.window_end": "窗口终点",
        "main.caption_ok": "分析口径：{pair}（看涨 {bull}）· 先看现价 → 自设概率区间 → 再运行蒙特卡洛",
        "main.caption_need": "最高日高分档 · 七步情报流水线 · 请先完成侧栏「开始设置」",
        # Label audit
        "audit.title": "证据人工标注",
        "audit.caption": "对照模型方向填写你的判断；保存后可看同意率，也可用于重算权重。无证据时仍可展开「怎么填？」并加载练习样例。",
        "audit.how": "怎么填？",
        "audit.allowed": "允许值 — 方向：{dirs}｜agree：{agrees}｜类别见下拉框（与模型词表一致）",
        "audit.no_rows": "**没有真实证据时，请先加载练习样例练手：**",
        "audit.load_demo": "加载练习样例（演示标注）",
        "audit.demo_info": "当前为**练习样例**（非本次运行证据）。标完可点下方「退出练习」回到真实列表。",
        "audit.exit_demo": "退出练习样例",
        "audit.prefill": "一键按模型预填再改",
        "audit.prefill.help": "把 model_* 复制到 human_*，再只改不同意的",
        "audit.save": "保存标注到 output/",
        "audit.clear": "清空人工列",
        "audit.open_link": "打开链接",
        "audit.model_ro": "模型只读 — 方向：`{md}`（{md_zh}）｜类别：{cat}",
        "audit.human_dir": "human_direction（你的方向）",
        "audit.human_cat": "human_category（你的类别）",
        "audit.agree": "agree（可自动）",
        "audit.unset": "（未选）",
        "audit.filled": "已填方向 {n}/{total} · 保存路径：`{path}`",
        "audit.spot_rate": "抽检准确率",
        "audit.spot_rate.cap": "= 同意率 yes/(yes+no)",
        "audit.jump_banner": "**↓ 证据人工标注在下方**",
        "audit.jump_link": "[跳转到标注区](#label-audit-section)",
        "audit.jump_full": "**↓ 证据人工标注在下方**（紧随本审计面板；完整报告 / PDF 在更下面。侧栏也可打开 ⑨。） · [跳转到标注区](#label-audit-section)",
        "val.use": "使用",
        "val.not_use": "不使用",
        "val.review": "人工确认",
        "val.skip": "自动跳过",
    },
    LANG_EN: {
        "lang.label": "UI language",
        "lang.help": "Applies immediately; stored in ?lang= so reload keeps your choice.",
        "report.lang": "Report language",
        "report.lang.help": "Language for PDF / HTML / Markdown body. Default is bilingual (ZH+EN together): news/MC run once, templates render twice. Independent of UI language; source quotes stay original.",
        "report.lang.both": "Chinese + English together",
        "report.stance_llm": "Also use LLM for reference summaries on historical runs",
        "report.stance_llm.help": "Historical cheap path defaults to extractive summaries (~$0). Enable to batch DeepSeek (~20 refs/call).",
        "side.cost": "Per-report cost estimate",
        "auth.caption": "Enter the access password (override default via APP_PASSWORD / FX_REPORT_PASSWORD).",
        "auth.password": "Password",
        "auth.enter": "Enter",
        "auth.wrong": "Wrong password",
        "auth.empty": "Please enter the password",
        "dlg.start_title": "Start setup",
        "dlg.missing_title": "Can't run yet",
        "dlg.setup_mode": "Setup mode",
        "dlg.setup_mode.help": "Simple: after pair + bullish, the system recommends peak engine / jumps / VR / clustering / calibration / human review. Expert: you must pick these explicitly—no silent defaults.",
        "dlg.simple.caption": "Simple mode: only pair + bullish currency (optional order PDF/image). Algorithms are recommended from calib JSON → engine compare → product defaults. Switch to Expert from the sidebar afterward.",
        "dlg.expert.caption": "Expert mode: pick peak engine / calibration / human review yourself (no pre-selection). Order PDF/image can still fill pair + bullish; algorithm fields stay manual.",
        "dlg.upload_pdf": "Upload order PDF/image (optional)",
        "dlg.upload_pdf.label": "Upload order PDF/image",
        "dlg.upload_pdf.help": "Accepts PDF, JPEG/JPG, PNG. Detects pair, bullish side, Barrier/Strike/buckets; on failure fill manually—fields are never invented.",
        "dlg.pdf.parsing": "Parsing order…",
        "dlg.pdf.fail": "Could not parse the order—please fill manually.",
        "dlg.pdf.ok": "Order parsed (please verify prefilled fields; unrecognized items still need a manual choice).",
        "dlg.pair_mode": "Pair mode (required)",
        "dlg.pair_mode.hint": "Choose Catalog or Custom first.",
        "dlg.pair": "Currency pair (required)",
        "dlg.pair_custom": "BASE/QUOTE (required, e.g. EUR/USD)",
        "dlg.pair_custom.ph": "Enter a pair, e.g. EUR/USD",
        "dlg.ticker": "Internal ticker (optional; default strips slash)",
        "dlg.bullish": "Bullish currency (required)",
        "dlg.bullish.help": "Bullish currency strengthening = higher analysis quote. Choosing quote flips the analysis pair.",
        "dlg.bullish.wait": "Select a pair first, then the bullish currency.",
        "dlg.algo.preview": "Algorithms recommended for this run",
        "dlg.algo.wait": "After you pick a pair, the recommended algorithm set will appear here.",
        "dlg.to_expert": "Switch to Expert",
        "dlg.peak_engine": "Peak engine (required)",
        "dlg.peak_engine.help": "path_max = discrete GBM + Merton path max; brownian_bridge = continuous max via reflection (no jumps)",
        "dlg.calibrated": "Use Stage-1 calibrated params (required)",
        "dlg.calibrated.help": "Use: prefer output/ then bundled JSON. Don't use: default priors. Never auto-checked just because a file exists.",
        "dlg.human_review": "Human review for uncertain evidence (required)",
        "dlg.human_review.help": "Need review = pause on low-confidence evidence; Auto-skip = don't interrupt the pipeline.",
        "dlg.confirm": "Confirm start setup",
        "dlg.cancel": "Cancel",
        "dlg.algo_fail": "Could not build a recommendation—select a currency pair first.",
        "dlg.missing.caption": "Complete the required choices, then click Run analysis. Bucket boundary mode is set in the main Probability bands section.",
        "dlg.open_start": "Open start setup",
        "dlg.got_it": "Got it",
        "opt.simple": "Simple (recommended)",
        "opt.expert": "Expert",
        "opt.catalog": "Catalog",
        "opt.custom": "Custom",
        "opt.use": "Use",
        "opt.not_use": "Don't use",
        "opt.need_review": "Need human review",
        "opt.auto_skip": "Auto-skip",
        "opt.cal_use": "Use",
        "opt.cal_skip": "Don't use",
        "missing.prefix": "You still need to choose: ",
        "missing.empty": "You still need to choose: required fields",
        "placeholder.choice": PLACEHOLDER_EN,
        "side.lang": "Language / 语言",
        "side.start": "① Start setup",
        "side.start.empty": "Start setup not finished yet (pair / bullish currency, etc.).",
        "side.start.summary": "**{pair}** · bullish **{bull}**  \nMode `{mode}` · peak `{eng}` · calib {cal} · uncertain {hr}",
        "side.start.algo_caption": "Algorithms were auto-recommended (see audit on the results page).",
        "side.to_expert": "Switch to Expert",
        "side.to_expert.help": "Open start setup in Expert mode to pick peak engine / calibration / HITL by hand.",
        "side.open_start": "Open start setup…",
        "side.start.invalid": "Invalid start setup: {exc}",
        "side.toc": "Sidebar: ① Start → ② Fetch → ③ Monte Carlo → ④ Mapping → ⑤ Scenarios → ⑥ Evidence → ⑦ Rubric → ⑧ Sources｜bucket cuts are in the main area",
        "side.fetch": "② Fetch & classify",
        "side.mc": "③ Monte Carlo",
        "side.map": "④ Evidence → params",
        "side.scenario": "⑤ Scenario priors",
        "side.evidence": "⑥ Template evidence scorecard",
        "side.rubric": "⑦ Strength rubric",
        "side.sources": "⑧ Data source status",
        "side.label_audit": "⑨ Evidence labeling",
        "side.setup_gap": "Setup gap (news keys)",
        "side.setup_gap.body": (
            "`NEWSAPI_KEY` / `FINNHUB_API_KEY` not detected. "
            "Without them the app still tries central-bank RSS + Google News; References stay thin. "
            "DeepSeek alone does **not** invent links. See `docs/deploy-docker.md`."
        ),
        "side.label_audit.has_run": "Labeling lives in the main area **just under “This-run audit”** (above the full report—no need to scroll to the bottom).",
        "side.label_audit.n": "Evidence rows: {n}",
        "side.label_audit.demo_hint": " · with no evidence, load the practice demo",
        "side.label_audit.jump": "[↓ Jump to labeling](#label-audit-section)",
        "side.label_audit.need_run": "Click Run analysis in the main area first. The labeling block appears under the audit panel; even with no news evidence you’ll see How to fill + Load practice demo.",
        "side.label_audit.learn_progress": "Optional: live labels {n}/{need} (then enable learned strength)",
        "side.label_audit.learn_ready": "Label learning ready ({n} labels)",
        "side.cal.use": "Calibration: Start setup chose Use",
        "side.cal.skip": "Calibration: Start setup chose Don't use",
        "side.cal.unset": "Calibration: not chosen in Start setup yet",
        "main.need_start": "Finish Start setup (sidebar ①) first: pair, bullish currency, peak engine, calibrated params, and human review for uncertain evidence.",
        "main.need_start_short": "Open Start setup in sidebar ① and pick pair + bullish currency. Spot and bucket settings appear right after.",
        "main.run": "Run analysis",
        "main.run.need_spot": "Need a successful spot quote first",
        "main.compare": "Dual-engine compare",
        "main.compare.help": "path_max vs brownian_bridge (downsampled sims, MC only)",
        "main.no_spot": "Can't run without spot (buckets and MC both need the analysis quote).",
        "main.ready_hint": "Confirm Start setup and probability bands, then Run analysis. Sidebar ②–⑧ tune fetch/model; API keys may be empty.",
        "main.rerun": "Run again",
        "main.rerun.help": "Re-run with the current Start setup and bands; keep the prior PDF/HTML download until the new result replaces it.",
        "main.new_report": "New report",
        "main.new_report.help": "Open Start setup to change pair/bullish etc.; prior report downloads stay available.",
        "main.after_report_hint": "A report is ready. Click Run again or New report, or use Run analysis above.",
        "main.window_start": "Window start",
        "main.window_end": "Window end",
        "main.caption_ok": "Analysis quote: {pair} (bullish {bull}) · spot → bands → Monte Carlo",
        "main.caption_need": "Peak-bucket · seven-step pipeline · finish Start setup in the sidebar first",
        "audit.title": "Evidence labeling",
        "audit.caption": "Enter your judgment vs the model; after save you can see agreement and optionally reweight. With no evidence you can still open How to fill and load a practice demo.",
        "audit.how": "How to fill?",
        "audit.allowed": "Allowed — direction: {dirs}｜agree: {agrees}｜categories in the dropdown (same lexicon as the model)",
        "audit.no_rows": "**No live evidence yet—load a practice demo first:**",
        "audit.load_demo": "Load practice demo",
        "audit.demo_info": "You are on the **practice demo** (not this run’s evidence). Use Exit practice below to return to the live list.",
        "audit.exit_demo": "Exit practice demo",
        "audit.prefill": "Prefill from model, then edit",
        "audit.prefill.help": "Copy model_* into human_*, then change only disagreements",
        "audit.save": "Save labels to output/",
        "audit.clear": "Clear human columns",
        "audit.open_link": "Open link",
        "audit.model_ro": "Model (read-only) — direction: `{md}` ({md_zh})｜category: {cat}",
        "audit.human_dir": "human_direction (yours)",
        "audit.human_cat": "human_category (yours)",
        "audit.agree": "agree (auto-ok)",
        "audit.unset": "(unset)",
        "audit.filled": "Directions filled {n}/{total} · save path: `{path}`",
        "audit.spot_rate": "Spot-check accuracy",
        "audit.spot_rate.cap": "= agreement yes/(yes+no)",
        "audit.jump_banner": "**↓ Evidence labeling below**",
        "audit.jump_link": "[Jump to labeling](#label-audit-section)",
        "audit.jump_full": "**↓ Evidence labeling below** (right under this audit; full report / PDF further down. Sidebar ⑨ also links here.) · [Jump to labeling](#label-audit-section)",
        "val.use": "Use",
        "val.not_use": "Don't use",
        "val.review": "Human review",
        "val.skip": "Auto-skip",
    },
}


def normalize_lang(value: object | None) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if raw in ("en", "en-us", "en-gb", "english"):
        return LANG_EN
    if raw in ("zh", "zh-cn", "zh-hans", "cn", "chinese", "中文"):
        return LANG_ZH
    return DEFAULT_LANG


def start_field_label(key: str, lang: str | None = None) -> str:
    lang = normalize_lang(lang or DEFAULT_LANG)
    table = START_FIELD_LABELS.get(lang) or START_FIELD_LABELS[LANG_ZH]
    return table.get(key, START_FIELD_LABELS[LANG_ZH].get(key, key))


def choice_placeholder(lang: str | None = None) -> str:
    return t("placeholder.choice", lang=lang)


def t(key: str, *, lang: str | None = None, **kwargs: Any) -> str:
    """Translate ``key`` for ``lang`` (default: current / zh). Supports ``{name}`` format."""
    lang = normalize_lang(lang if lang is not None else _current_lang_or_default())
    table = _STRINGS.get(lang) or _STRINGS[LANG_ZH]
    text = table.get(key)
    if text is None:
        text = _STRINGS[LANG_ZH].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def _current_lang_or_default() -> str:
    try:
        import streamlit as st

        if LANG_STATE_KEY in st.session_state:
            return normalize_lang(st.session_state.get(LANG_STATE_KEY))
    except Exception:
        pass
    return DEFAULT_LANG


def get_lang() -> str:
    """Return active UI language (session → query → zh)."""
    try:
        import streamlit as st

        if LANG_STATE_KEY in st.session_state:
            return normalize_lang(st.session_state.get(LANG_STATE_KEY))
        qp = _query_lang()
        if qp:
            return qp
    except Exception:
        pass
    return DEFAULT_LANG


def _query_lang() -> str | None:
    try:
        import streamlit as st

        raw = st.query_params.get(LANG_QUERY_KEY)
        if raw is None or raw == "":
            return None
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
        if raw is None or raw == "":
            return None
        return normalize_lang(raw)
    except Exception:
        return None


def init_language() -> str:
    """
    Ensure session_state has ui_lang; seed from ?lang= when present.
    Call once near the top of each script run (before widgets that need t()).
    """
    import streamlit as st

    qp = _query_lang()
    if LANG_STATE_KEY not in st.session_state:
        st.session_state[LANG_STATE_KEY] = qp or DEFAULT_LANG
    elif qp and normalize_lang(qp) != normalize_lang(st.session_state.get(LANG_STATE_KEY)):
        # URL wins when user opens a shared link with ?lang=
        st.session_state[LANG_STATE_KEY] = normalize_lang(qp)
    lang = normalize_lang(st.session_state[LANG_STATE_KEY])
    st.session_state[LANG_STATE_KEY] = lang
    try:
        if st.query_params.get(LANG_QUERY_KEY) != lang:
            st.query_params[LANG_QUERY_KEY] = lang
    except Exception:
        pass
    return lang


def set_language(lang: str) -> str:
    """Persist language to session_state + query param."""
    import streamlit as st

    lang = normalize_lang(lang)
    st.session_state[LANG_STATE_KEY] = lang
    try:
        st.query_params[LANG_QUERY_KEY] = lang
    except Exception:
        pass
    return lang


def render_language_selector(*, location: str = "sidebar", key: str = "ui_lang_select") -> str:
    """
    Prominent language toggle. ``location``: ``sidebar`` | ``main``.
    Returns the selected language code.
    """
    import streamlit as st

    lang = init_language()
    options = list(SUPPORTED_LANGS)
    labels = [LANG_LABELS[c] for c in options]
    idx = options.index(lang) if lang in options else 0
    host = st.sidebar if location == "sidebar" else st
    pick = host.radio(
        t("lang.label", lang=lang) if location == "main" else t("side.lang", lang=lang),
        options,
        index=idx,
        format_func=lambda c: LANG_LABELS.get(c, c),
        horizontal=True,
        key=key,
        help=t("lang.help", lang=lang),
    )
    pick = normalize_lang(pick)
    if pick != lang:
        set_language(pick)
        st.rerun()
    return pick


def format_missing_message(missing_labels: list[str] | tuple[str, ...], *, lang: str | None = None) -> str:
    lang = normalize_lang(lang if lang is not None else _current_lang_or_default())
    labels = [str(x).strip() for x in missing_labels if str(x).strip()]
    if not labels:
        return t("missing.empty", lang=lang)
    return t("missing.prefix", lang=lang) + ("、" if lang == LANG_ZH else ", ").join(labels)


def localize_start_labels(keys: Mapping[str, object] | None = None, *, lang: str | None = None) -> dict[str, str]:
    """Return {field_key: localized_label} for start-required fields."""
    lang = normalize_lang(lang if lang is not None else _current_lang_or_default())
    src = START_FIELD_LABELS.get(lang) or START_FIELD_LABELS[LANG_ZH]
    if keys is None:
        return dict(src)
    return {k: src.get(k, k) for k in keys}
