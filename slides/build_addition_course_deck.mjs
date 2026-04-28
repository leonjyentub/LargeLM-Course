import path from "node:path";
import { pathToFileURL } from "node:url";

const runtimeNodeModules =
  process.env.CODEX_NODE_MODULES ||
  "C:/Users/leonjye/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const artifactUrl = pathToFileURL(
  path.join(runtimeNodeModules, "@oai/artifact-tool/dist/artifact_tool.mjs"),
).href;

const {
  Presentation,
  PresentationFile,
  column,
  row,
  grid: rawGrid,
  panel,
  text,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} = await import(artifactUrl);

const OUT = "C:/Users/leonjye/Documents/PythonProjs/LargeLM-Course/slides/output/addition_nlp_course.pptx";
const W = 1920;
const H = 1080;
const C = {
  ink: "#17212B",
  muted: "#5F6B76",
  paper: "#F7F4ED",
  blue: "#245E9D",
  cyan: "#2AA7A5",
  red: "#C34A36",
  gold: "#D49A2A",
  green: "#3F7F55",
  dark: "#102A3A",
  softBlue: "#DDEBFA",
  softCyan: "#DDF3F1",
  softGold: "#F5E7C8",
  softRed: "#F3D8D2",
  softGreen: "#DDEBDD",
  white: "#FFFFFF",
};

const deck = Presentation.create({ slideSize: { width: W, height: H } });

function grid(opts, children) {
  const colCount = Math.max(1, opts.columns?.length ?? children.length);
  const rows = [];
  for (let i = 0; i < children.length; i += colCount) {
    rows.push(row(
      { width: fill, height: opts.rows?.length === 1 ? fill : hug, gap: opts.columnGap ?? 20 },
      children.slice(i, i + colCount),
    ));
  }
  return column({ width: opts.width ?? fill, height: opts.height ?? fill, gap: opts.rowGap ?? 20 }, rows);
}

function t(value, size = 28, color = C.ink, extra = {}) {
  return text(value, {
    width: extra.width ?? fill,
    height: extra.height ?? hug,
    style: {
      fontSize: size,
      color,
      bold: extra.bold ?? false,
      fontFace: "Microsoft JhengHei",
      ...extra.style,
    },
    ...extra.options,
  });
}

function chip(label, color = C.blue) {
  return panel(
    { width: hug, height: hug, padding: { x: 18, y: 8 }, fill: color, borderRadius: 16 },
    t(label, 18, C.white, { bold: true, width: hug }),
  );
}

function card(title, body, color = C.softBlue) {
  return panel(
    { width: fill, height: fill, padding: { x: 28, y: 24 }, fill: color, borderRadius: 8 },
    column({ width: fill, height: fill, gap: 14 }, [
      t(title, 27, C.ink, { bold: true }),
      t(body, 22, C.muted, { width: fill }),
    ]),
  );
}

function slide(title, subtitle, body, accent = C.blue) {
  const s = deck.slides.add();
  s.compose(
    column({ name: "root", width: fill, height: fill, padding: { x: 86, y: 68 }, gap: 30, fill: C.paper }, [
      row({ width: fill, height: hug, gap: 18 }, [chip("NLP Addition", accent), chip("PyTorch", C.dark)]),
      column({ width: fill, height: hug, gap: 18 }, [
        t(title, 54, C.ink, { bold: true, width: wrap(1500) }),
        subtitle ? t(subtitle, 26, C.muted, { width: wrap(1360) }) : null,
      ].filter(Boolean)),
      rule({ width: fixed(260), stroke: accent, weight: 5 }),
      body,
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

slide(
  "把加法當成自然語言任務",
  "同一份 CSV，切成分類、迴歸、多標籤、序列生成，讓學生看見 loss 與輸出設計如何改變模型學到的東西。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1)], columnGap: 26 }, [
    card("輸入", "`166+644` 是字元序列，不給模型數學規則。", C.softBlue),
    card("目標", "答案 810 可以是 class id、連續值、四個 digit，或生成序列。", C.softCyan),
    card("核心問題", "模型到底是在記憶分布，還是在學位值、進位與輸出格式？", C.softGold),
  ]),
  C.blue,
);

slide(
  "資料集格式與切分",
  "data/addition/*.csv 與 data/addition_smoke/*.csv 都使用相同欄位，smoke 版適合課堂快速測試。",
  column({ width: fill, height: fill, gap: 28 }, [
    panel({ width: fill, height: hug, padding: { x: 30, y: 22 }, fill: C.white, borderRadius: 8 },
      t("a,b,expr,sum\n166,644,166+644,810\n453,932,453+932,1385", 28, C.ink, { style: { fontFace: "Consolas" } })),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1)], columnGap: 24 }, [
      card("train", "學習字元模式與答案對應。", C.softGreen),
      card("val", "選 best_model.pt，避免只看訓練集。", C.softGold),
      card("test", "輸出 metrics.json，供 compare_all_methods.py 彙整。", C.softBlue),
    ]),
  ]),
  C.cyan,
);

slide(
  "共同前處理：從字串到 token id",
  "common/data_utils.py 把加號與數字都視為字元 token，padding 讓 batch 形狀固定。",
  grid({ width: fill, height: fill, columns: [fr(0.9), fr(1.1)], rows: [fr(1)], columnGap: 38 }, [
    column({ width: fill, height: fill, gap: 22 }, [
      card("INPUT_VOCAB", "<pad>=0, +=1, 0..9 對應字元 id。", C.softBlue),
      card("MAX_INPUT_LEN=7", "最大輸入是 `999+999`，短式子補 pad。", C.softCyan),
      card("Embedding", "所有模型先把 token id 轉成可學習向量。", C.softGold),
    ]),
    panel({ width: fill, height: fill, padding: { x: 34, y: 30 }, fill: C.dark, borderRadius: 8 },
      column({ width: fill, height: fill, gap: 22 }, [
        t("例：100+20", 34, C.white, { bold: true }),
        t("[2, 3, 3, 1, 4, 3, 0]", 34, "#BDE7E6", { style: { fontFace: "Consolas" } }),
        t("pad mask 讓 attention 或 pooling 忽略補齊位置。", 24, "#D7E5EA"),
      ])),
  ]),
  C.cyan,
);

slide(
  "輸出設計一：多元分類",
  "把每個可能的加總視為一個類別，使用 CrossEntropyLoss。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1)], columnGap: 34 }, [
    column({ width: fill, height: fill, gap: 22 }, [
      card("FFNNClassifier", "攤平成固定長度向量，示範最簡單 baseline。", C.softBlue),
      card("LSTM / GRU Classifier", "用最後 hidden state 表示整個運算式。", C.softCyan),
      card("SelfAttentionClassifier", "用 attention 看不同字元位置的關係。", C.softGold),
    ]),
    card("教學重點", "輸出維度是 2000；label 是 sum 的整數值。優點是訓練簡單，缺點是 `1998` 與 `1997` 沒有 digit 結構關係，模型較像在學 class boundary。", C.white),
  ]),
  C.blue,
);

slide(
  "輸出設計二：迴歸",
  "新增 train_lstm_regression.py：把答案縮放到 0..1，用 MSELoss 學連續數值。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1)], columnGap: 26 }, [
    card("target", "sum / 1998.0", C.softGreen),
    card("loss", "MSELoss，比較預測值與連續答案。", C.softGold),
    card("後處理", "乘回 1998、round、clamp，再算 exact_match、MAE、RMSE。", C.softRed),
  ]),
  C.green,
);

slide(
  "輸出設計三：多標籤 / 多位置分類",
  "LSTMMultiLabelClassifier 同時預測千、百、十、個四個 digit。",
  grid({ width: fill, height: fill, columns: [fr(1.1), fr(0.9)], rows: [fr(1)], columnGap: 34 }, [
    panel({ width: fill, height: fill, padding: { x: 34, y: 30 }, fill: C.white, borderRadius: 8 },
      column({ width: fill, height: fill, gap: 22 }, [
        t("810 -> 0810", 42, C.ink, { bold: true, style: { fontFace: "Consolas" } }),
        t("logits shape = [batch, 4, 10]", 30, C.blue, { style: { fontFace: "Consolas" } }),
        t("每個位置各做一次 CrossEntropyLoss，再加總。", 26, C.muted),
      ])),
    column({ width: fill, height: fill, gap: 22 }, [
      card("優點", "答案有 digit 結構，輸出空間從 2000 class 變成 4x10。", C.softCyan),
      card("限制", "四個位置是平行輸出，不會自然表達『上一位的生成結果』。", C.softGold),
    ]),
  ]),
  C.gold,
);

slide(
  "輸出設計四：左到右生成式",
  "LSTMSeq2Seq / TransformerSeq2Seq 使用 teacher forcing，逐步輸出答案 token。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1)], columnGap: 34 }, [
    card("decoder input", "<sos> 0 8 1 0", C.softBlue),
    card("decoder target", "0 8 1 0 <eos>", C.softCyan),
    card("loss", "CrossEntropyLoss(ignore_index=pad)，每個時間步都算。", C.softGold),
    card("inference", "greedy_decode：上一個預測 token 餵回 decoder。", C.softGreen),
  ]),
  C.cyan,
);

slide(
  "輸出設計五：右到左生成式",
  "新增 train_lstm_reverse_seq2seq.py：先生成個位數，讓輸出順序更接近直式加法的進位方向。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1)], columnGap: 34 }, [
    panel({ width: fill, height: fill, padding: { x: 34, y: 30 }, fill: C.dark, borderRadius: 8 },
      column({ width: fill, height: fill, gap: 22 }, [
        t("0810", 44, C.white, { bold: true, style: { fontFace: "Consolas" } }),
        t("左到右：0 -> 8 -> 1 -> 0", 29, "#BDE7E6", { style: { fontFace: "Consolas" } }),
        t("右到左：0 -> 1 -> 8 -> 0", 29, "#F7D98B", { style: { fontFace: "Consolas" } }),
      ])),
    column({ width: fill, height: fill, gap: 22 }, [
      card("資料差異", "encode_seq_target(total, reverse=True) 反轉 target digits。", C.softGold),
      card("評估差異", "decode 後先反轉回正常順序，再去除前導零比較 exact match。", C.softRed),
      card("討論題", "右到左是否更容易學會進位？可和左到右 seq2seq 對照。", C.softGreen),
    ]),
  ]),
  C.red,
);

slide(
  "模型結構比較",
  "把架構差異講成『資訊如何流動』，學生會比只記層名更容易理解。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 22, rowGap: 22 }, [
    card("FFNN", "固定位置記憶，沒有序列狀態。", C.softBlue),
    card("RNN", "LSTM/GRU 壓縮成最後 hidden state。", C.softCyan),
    card("Attention", "位置之間直接互看，pad mask 很重要。", C.softGold),
    card("Regression", "答案是連續值，容易展示尺度與誤差。", C.softGreen),
    card("Multi-label", "答案被拆成四個 digit 任務。", C.softRed),
    card("Seq2Seq", "encoder-decoder，輸出可長可短，最接近生成式 NLP。", C.white),
  ]),
  C.dark,
);

slide(
  "Loss 與 metric 的配對",
  "同一題換輸出形式，loss、後處理、評估指標都要跟著換。",
  grid({ width: fill, height: fill, columns: [fr(1.2), fr(1), fr(1)], rows: [auto, auto, auto, auto, auto], rowGap: 12, columnGap: 16 }, [
    t("任務", 24, C.ink, { bold: true }), t("Loss", 24, C.ink, { bold: true }), t("主要 metric", 24, C.ink, { bold: true }),
    t("多元分類", 23), t("CrossEntropy", 23), t("accuracy / macro_f1", 23),
    t("迴歸", 23), t("MSE", 23), t("MAE / RMSE / rounded exact", 23),
    t("多標籤 digit", 23), t("4 個 CrossEntropy 加總", 23), t("exact_match / digit_accuracy", 23),
    t("生成式", 23), t("token CrossEntropy", 23), t("exact_match / char_accuracy", 23),
  ]),
  C.green,
);

slide(
  "程式輸出契約",
  "每個 train_<method>.py 都維持相同輸出結構，方便自動比較與課堂展示。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1)], columnGap: 34 }, [
    panel({ width: fill, height: fill, padding: { x: 34, y: 30 }, fill: C.white, borderRadius: 8 },
      t("outputs/<method>/\n  train.log\n  metrics.json\n  training_curves.png\n  best_model.pt\n  roc_curve.png  # multiclass only", 30, C.ink, { style: { fontFace: "Consolas" } })),
    column({ width: fill, height: fill, gap: 22 }, [
      card("compare_all_methods.py", "讀取所有 metrics.json，輸出總表與比較圖。", C.softBlue),
      card("run_all_experiments.py", "從 JSON/YAML config 產生全流程命令，支援 --dry-run。", C.softCyan),
    ]),
  ]),
  C.blue,
);

slide(
  "課堂實作流程建議",
  "先讓學生跑 smoke，再逐步切換方法，最後用 comparison 看設計選擇的代價。",
  column({ width: fill, height: fill, gap: 26 }, [
    card("1. 觀察資料", "讀 train.csv 的 expr/sum，確認這是字元任務，不是直接呼叫加法。", C.softBlue),
    card("2. 跑單一模型", "先跑 FFNN 或 LSTM classifier，理解 embedding、logits、CrossEntropy。", C.softCyan),
    card("3. 換輸出形式", "比較 regression、multilabel、seq2seq、reverse seq2seq。", C.softGold),
    card("4. 討論泛化", "把位數、資料量、輸出方向、loss 換掉，觀察哪些模型真的學到規則。", C.softGreen),
  ]),
  C.gold,
);

slide(
  "可延伸的比較題",
  "這些題目適合當作作業或期末 mini project，能自然連到 NLP 與 deep learning 的核心觀念。",
  grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 26, rowGap: 26 }, [
    card("資料外推", "訓練 1-3 位數，測 4 位數，檢查是否學到演算法。", C.softBlue),
    card("進位切片", "把 test 分成有進位/無進位、多次進位，做 error analysis。", C.softRed),
    card("Scheduled sampling", "降低 teacher forcing 與 inference 的落差。", C.softGold),
    card("Attention 可視化", "對 self-attention 或 transformer 畫出哪些位置被關注。", C.softCyan),
  ]),
  C.red,
);

await PresentationFile.exportPptx(deck).then((blob) => blob.save(OUT));
console.log(OUT);
