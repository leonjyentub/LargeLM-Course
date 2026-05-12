from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def logsumexp(vec: np.ndarray) -> float:
	# 數值穩定版 logsumexp：
	# logsumexp(a) = m + log(sum_i exp(a_i - m)), 其中 m = max(a)
	# 目的：避免直接 exp 導致 overflow / underflow。
	m = np.max(vec)
	return float(m + np.log(np.sum(np.exp(vec - m))))


def build_vocab(sentences: list[list[str]]) -> dict[str, int]:
	vocab = sorted({tok for sent in sentences for tok in sent})
	return {tok: i for i, tok in enumerate(vocab)}


def build_label_map(label_sequences: list[list[str]]) -> dict[str, int]:
	labels = sorted({lab for seq in label_sequences for lab in seq})
	return {lab: i for i, lab in enumerate(labels)}


def to_one_hot_features(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
	feats = np.zeros((len(tokens), len(vocab)), dtype=np.float64)
	for t, tok in enumerate(tokens):
		feats[t, vocab[tok]] = 1.0
	return feats


@dataclass
class CRFGrads:
	w: np.ndarray
	trans: np.ndarray
	start: np.ndarray
	end: np.ndarray


class LinearChainCRF:
	"""A minimal linear-chain CRF implemented with NumPy only.

	Model score:
	score(x, y) = sum_t emission[t, y_t] + start[y_0] + end[y_T]
				  + sum_t trans[y_{t-1}, y_t]

	繁體中文對照：
	- CRF 不直接做逐 token 的 softmax，而是對整條標籤序列 Y 打分。
	- Sequence Score (線性鏈 CRF 常見形式)：
	  Score(X, Y) = Σ_t Transition(y_{t-1}, y_t) + Emission(x, t, y_t)
	- 條件機率：
	  P(Y|X) = exp(Score(X, Y)) / Z(X)
	- Partition function：
	  Z(X) = Σ_{Y'} exp(Score(X, Y'))
	"""

	def __init__(self, num_features: int, num_labels: int, seed: int = 42):
		rng = np.random.default_rng(seed)
		self.num_features = num_features
		self.num_labels = num_labels

		# self.trans[i, j]：標籤 i -> 標籤 j 的轉移分數 (Transition Score)
		self.w = rng.normal(loc=0.0, scale=0.01, size=(num_features, num_labels))
		self.trans = rng.normal(loc=0.0, scale=0.01, size=(num_labels, num_labels))
		# 起始與結束狀態分數：START -> y_0、y_T -> END
		self.start = np.zeros(num_labels, dtype=np.float64)
		self.end = np.zeros(num_labels, dtype=np.float64)

	def emissions(self, x: np.ndarray) -> np.ndarray:
		# Emission Score E[t, j]：第 t 個 token 被標成 label j 的分數。
		# 這裡使用簡化版：x @ W。
		# 在實務上常由 BiLSTM / CNN / Transformer / BERT 輸出。
		return x @ self.w

	def forward_log(self, emissions: np.ndarray) -> tuple[np.ndarray, float]:
		# Forward Algorithm 目的：高效率計算 log Z(X)，避免列舉 K^T 條路徑。
		# 定義 alpha_t(j)：到位置 t 且當前標籤為 j 的所有路徑總分 (log-space)。
		# 遞迴：
		# alpha_t(j) = logsumexp_i(alpha_{t-1}(i) + A[i, j] + E[t, j])
		t_len, num_labels = emissions.shape
		alpha = np.full((t_len, num_labels), -np.inf, dtype=np.float64)
		# 初始化：alpha_0(j) = A[START, j] + E[0, j]
		alpha[0] = self.start + emissions[0]

		for t in range(1, t_len):
			for curr in range(num_labels):
				alpha[t, curr] = emissions[t, curr] + logsumexp(alpha[t - 1] + self.trans[:, curr])

		# 結束：log Z(X) = logsumexp_j(alpha_T(j) + A[j, END])
		log_z = logsumexp(alpha[-1] + self.end)
		return alpha, log_z

	def backward_log(self, emissions: np.ndarray) -> np.ndarray:
		# Backward Algorithm：計算 beta，後續可用於邊際機率與梯度。
		t_len, num_labels = emissions.shape
		beta = np.full((t_len, num_labels), -np.inf, dtype=np.float64)
		beta[-1] = self.end

		for t in range(t_len - 2, -1, -1):
			for prev in range(num_labels):
				beta[t, prev] = logsumexp(self.trans[prev, :] + emissions[t + 1, :] + beta[t + 1, :])
		return beta

	def sequence_score(self, emissions: np.ndarray, y: np.ndarray) -> float:
		# Gold sequence score：
		# Score(X, Y) = START[y_0] + Σ_t E[t, y_t] + Σ_{t>0} A[y_{t-1}, y_t] + END[y_T]
		score = self.start[y[0]] + emissions[0, y[0]]
		for t in range(1, len(y)):
			score += self.trans[y[t - 1], y[t]] + emissions[t, y[t]]
		score += self.end[y[-1]]
		return float(score)

	def nll_and_grads(self, x: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> tuple[float, CRFGrads]:
		# CRF 訓練核心：
		# log P(Y|X) = Score(X, Y) - log Z(X)
		# Loss (NLL) = -log P(Y|X) = log Z(X) - Score(X, Y)
		emissions = self.emissions(x)
		alpha, log_z = self.forward_log(emissions)
		beta = self.backward_log(emissions)

		gold_score = self.sequence_score(emissions, y)
		nll = log_z - gold_score

		t_len = emissions.shape[0]
		grads_w = np.zeros_like(self.w)
		grads_trans = np.zeros_like(self.trans)
		grads_start = np.zeros_like(self.start)
		grads_end = np.zeros_like(self.end)

		# Gold counts：正確序列的特徵計數 (負號代表在 NLL 梯度中要被拉高分數)。
		grads_start[y[0]] -= 1.0
		grads_end[y[-1]] -= 1.0
		for t in range(t_len):
			grads_w[:, y[t]] -= x[t]
			if t > 0:
				grads_trans[y[t - 1], y[t]] -= 1.0

		# Expected counts：模型分佈 p(Y|X) 下的期望特徵計數。
		# 透過 alpha/beta 邊際機率取得，與 Gold counts 相減即為梯度。
		log_node_marg = alpha + beta - log_z
		node_marg = np.exp(log_node_marg)

		grads_start += node_marg[0]
		grads_end += node_marg[-1]
		for t in range(t_len):
			grads_w += np.outer(x[t], node_marg[t])

		for t in range(1, t_len):
			# 轉移邊際機率：位置 t 時，i -> j 的機率。
			edge_log = (
				alpha[t - 1][:, None]
				+ self.trans
				+ emissions[t][None, :]
				+ beta[t][None, :]
				- log_z
			)
			grads_trans += np.exp(edge_log)

		# L2 正則化：限制權重過大，降低過擬合。
		nll += 0.5 * l2 * (np.sum(self.w**2) + np.sum(self.trans**2))
		grads_w += l2 * self.w
		grads_trans += l2 * self.trans

		grads = CRFGrads(w=grads_w, trans=grads_trans, start=grads_start, end=grads_end)
		return float(nll), grads

	def step(self, grads: CRFGrads, lr: float) -> None:
		self.w -= lr * grads.w
		self.trans -= lr * grads.trans
		self.start -= lr * grads.start
		self.end -= lr * grads.end

	def viterbi_decode(self, x: np.ndarray) -> list[int]:
		# Viterbi Decoding：找最佳路徑
		# Y* = argmax_Y Score(X, Y)
		# 與 Forward 不同：Forward 用 logsumexp 聚合所有路徑；Viterbi 用 max 找單一路徑。
		emissions = self.emissions(x)
		t_len, num_labels = emissions.shape

		dp = np.full((t_len, num_labels), -np.inf, dtype=np.float64)
		bp = np.zeros((t_len, num_labels), dtype=np.int64)

		dp[0] = self.start + emissions[0]
		for t in range(1, t_len):
			for curr in range(num_labels):
				# delta_t(j) = max_i(delta_{t-1}(i) + A[i, j] + E[t, j])
				scores = dp[t - 1] + self.trans[:, curr]
				best_prev = int(np.argmax(scores))
				dp[t, curr] = scores[best_prev] + emissions[t, curr]
				# backpointer_t(j) 記錄最佳前驅 i，最後可回溯完整路徑。
				bp[t, curr] = best_prev

		last = int(np.argmax(dp[-1] + self.end))
		path = [last]
		for t in range(t_len - 1, 0, -1):
			path.append(int(bp[t, path[-1]]))
		path.reverse()
		return path


def encode_labels(seq: list[str], label2id: dict[str, int]) -> np.ndarray:
	return np.asarray([label2id[y] for y in seq], dtype=np.int64)


def print_transition_matrix(trans: np.ndarray, id2label: dict[int, str]) -> None:
	print("\n=== Trained Transition Matrix (A) ===")
	labels = [id2label[i] for i in range(len(id2label))]
	print("Rows = prev label, Cols = curr label")
	print("labels:", labels)

	header = "prev\\curr" + "".join(f"\t{lab}" for lab in labels)
	print(header)
	for i, prev_lab in enumerate(labels):
		row_vals = "".join(f"\t{trans[i, j]:.4f}" for j in range(len(labels)))
		print(f"{prev_lab}{row_vals}")


def print_emission_matrix(w: np.ndarray, vocab: dict[str, int], id2label: dict[int, str]) -> None:
	print("\n=== Trained Emission Weight Matrix (W) ===")
	print("Rows = token, Cols = label")

	id2token = {idx: tok for tok, idx in vocab.items()}
	labels = [id2label[i] for i in range(len(id2label))]
	print("labels:", labels)

	header = "token" + "".join(f"\t{lab}" for lab in labels)
	print(header)
	for i in range(w.shape[0]):
		tok = id2token[i]
		row_vals = "".join(f"\t{w[i, j]:.4f}" for j in range(w.shape[1]))
		print(f"{tok}{row_vals}")


def train_crf_demo(epochs: int = 60, lr: float = 0.2, seed: int = 7) -> None:
	# Sequence Labeling 任務：
	# 給定輸入序列 x=[x1,...,xn]，預測對應標籤序列 y=[y1,...,yn]。
	# 這裡使用簡化 NER toy data 做訓練示範。
	train_sentences = [
		["John", "lives", "in", "New", "York"],
		["Mary", "works", "at", "Google"],
		["Google", "is", "in", "California"],
		["John", "works", "at", "OpenAI"],
		["Alice", "lives", "in", "California"],
		["Bob", "works", "at", "Microsoft"],
		["Microsoft", "is", "in", "Seattle"],
		["Alice", "visited", "New", "York"],
		["Bob", "lives", "in", "Seattle"],
		["OpenAI", "is", "in", "San", "Francisco"],
		["Mary", "visited", "California"],
		["John", "visited", "San", "Francisco"],
		["Bob", "works", "at", "OpenAI"],
		["Google", "is", "in", "San", "Francisco"],
		["Alice", "works", "at", "Microsoft"],
		#一個長一點的範例，有and連接詞，測試模型對較長序列的處理能力。
		["Alice", "visited", "Microsoft", "and", "then", "went", "to", "Seattle"],
	]
	train_labels = [
		["B-PER", "O", "O", "B-LOC", "I-LOC"],
		["B-PER", "O", "O", "B-ORG"],
		["B-ORG", "O", "O", "B-LOC"],
		["B-PER", "O", "O", "B-ORG"],
		["B-PER", "O", "O", "B-LOC"],
		["B-PER", "O", "O", "B-ORG"],
		["B-ORG", "O", "O", "B-LOC"],
		["B-PER", "O", "B-LOC", "I-LOC"],
		["B-PER", "O", "O", "B-LOC"],
		["B-ORG", "O", "O", "B-LOC", "I-LOC"],
		["B-PER", "O", "B-LOC"],
		["B-PER", "O", "B-LOC", "I-LOC"],
		["B-PER", "O", "O", "B-ORG"],
		["B-ORG", "O", "O", "B-LOC", "I-LOC"],
		["B-PER", "O", "O", "B-ORG"],
		["B-PER", "O", "B-ORG", "O", "O", "O", "O", "B-LOC"],
	]

	test_sentences = [
		["Mary", "lives", "in", "New", "York"],  
		["Alice", "works", "at", "OpenAI"],
		["Bob", "works", "at", "Google"],
		["Microsoft", "is", "in", "California"],
		["John", "visited", "Seattle"],
		["OpenAI", "is", "in", "New", "York"],
		#長一點的範例
		["Alice", "visited", "San", "Francisco", "and", "then", "went", "to", "San", "Diego"],
	]

	vocab = build_vocab(train_sentences + test_sentences)
	label2id = build_label_map(train_labels)
	id2label = {v: k for k, v in label2id.items()}

	x_train = [to_one_hot_features(sent, vocab) for sent in train_sentences]
	y_train = [encode_labels(seq, label2id) for seq in train_labels]

	crf = LinearChainCRF(num_features=len(vocab), num_labels=len(label2id), seed=seed)

	rng = np.random.default_rng(seed)
	for epoch in range(1, epochs + 1):
		order = rng.permutation(len(x_train))
		total_nll = 0.0

		for idx in order:
			nll, grads = crf.nll_and_grads(x_train[idx], y_train[idx], l2=1e-4)
			crf.step(grads, lr=lr)
			total_nll += nll

		if epoch % 10 == 0 or epoch == 1:
			print(f"Epoch {epoch:02d} | avg NLL = {total_nll / len(x_train):.4f}")

	# 輸出訓練後參數，並附上欄位對應。
	print_transition_matrix(crf.trans, id2label)
	print_emission_matrix(crf.w, vocab, id2label)

	def decode_and_print(sentences: list[list[str]], title: str) -> None:
		print(f"\n{title}")
		for sent in sentences:
			x = to_one_hot_features(sent, vocab)
			pred_ids = crf.viterbi_decode(x)
			pred = [id2label[i] for i in pred_ids]
			print("Sentence:", " ".join(sent))
			print("Pred    :", " ".join(pred))

	decode_and_print(train_sentences, "Training set predictions")
	decode_and_print(test_sentences, "Test set predictions")


if __name__ == "__main__":
	train_crf_demo()
