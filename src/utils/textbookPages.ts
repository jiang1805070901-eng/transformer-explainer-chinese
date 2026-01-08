import { get } from 'svelte/store';
import {
	expandedBlock,
	weightPopover,
	isBoundingBoxActive,
	textbookCurrentPageId,
	isExpandOrCollapseRunning,
	isFetchingModel,
	userId
} from '~/store';
import {
	highlightElements,
	removeHighlightFromElements,
	applyTransformerBoundingHeight,
	resetElementsHeight,
	highlightAttentionPath,
	removeAttentionPathHighlight,
	removeFingerFromElements
} from '~/utils/textbook';
import { drawResidualLine } from './animation';

export interface TextbookPage {
	id: string;
	title: string;
	content?: string;
	component?: any;
	timeoutId?: number;
	on: () => void;
	out: () => void;
	complete?: () => void;
}

const { drawLine, removeLine } = drawResidualLine();

export const textPages: TextbookPage[] = [
	{
		id: 'what-is-transformer',
		title: '什么是 Transformer?',
		content: `<p><strong>Transformer</strong> 是现代 AI 的核心架构,为 ChatGPT 和 Gemini 等模型提供支持。它于 2017 年推出,彻底改变了 AI 处理信息的方式。相同的架构既用于在海量数据集上训练,也用于推理生成输出。这里我们使用 GPT-2 (small),它比新版本更简单,但非常适合学习基础知识。</p>
`,
		on: () => {},
		out: () => {}
	},
	{
		id: 'how-transformers-work',
		title: 'Transformer 如何工作?',
		content: `<p>Transformer 并不神奇——它们通过逐步构建文本来回答:</p>
	<blockquote class="question">
		"跟在这个输入后面最可能的下一个词是什么?"
	</blockquote>
	<p>在这里我们探索训练好的模型如何生成文本。编写你自己的文本或使用示例,然后点击<strong>生成</strong>来查看它的运行。如果模型还没准备好,请尝试另一个<strong>示例</strong>。</p>`,
		on: () => {
			highlightElements(['.input-form']);
			if (get(isFetchingModel)) {
				highlightElements(['.input-form .select-button']);
			} else {
				highlightElements(['.input-form .generate-button']);
			}
		},
		out: () => {
			removeHighlightFromElements([
				'.input-form',
				'.input-form .select-button',
				'.input-form .generate-button'
			]);
		},
		complete: () => {
			removeFingerFromElements(['.input-form .select-button', '.input-form .generate-button']);
			if (get(textbookCurrentPageId) === 'how-transformers-work') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'how-transformers-work'
				});
			}
		}
	},
	{
		id: 'transformer-architecture',
		title: 'Transformer 架构',
		content:
			'<p>Transformer 有三个主要部分:</p><div class="numbered-list"><div class="numbered-item"><span class="number-circle">1</span><div class="item-content"><strong>嵌入</strong>将文本转换为数字。</div></div><div class="numbered-item"><span class="number-circle">2</span><div class="item-content"><strong>Transformer 块</strong>通过自注意力混合信息,并通过 MLP 进行细化。</div></div><div class="numbered-item"><span class="number-circle">3</span><div class="item-content"><strong>概率</strong>确定每个下一个 token 的可能性。</div></div></div>',
		on: () => {
			const selectors = [
				'.step.embedding',
				'.step.softmax',
				'.transformer-bounding',
				'.transformer-bounding-title'
			];
			highlightElements(selectors);
			applyTransformerBoundingHeight(['.softmax-bounding', '.embedding-bounding']);
		},
		out: () => {
			const selectors = [
				'.step.embedding',
				'.step.softmax',
				'.transformer-bounding',
				'.transformer-bounding-title'
			];
			removeHighlightFromElements(selectors);
			resetElementsHeight(['.softmax-bounding', '.embedding-bounding']);
		}
	},
	{
		id: 'embedding',
		title: '嵌入',
		content: `<p>在 Transformer 可以使用文本之前,它首先将文本分解成小单元,并将每个单元表示为一个数字列表(向量)。这个过程称为<strong>嵌入</strong>,该术语既可以指过程本身,也可以指生成的向量。</p><p>在此工具中,每个向量显示为一个矩形,将鼠标悬停在上面会显示其大小。</p>`,
		on: () => {
			highlightElements(['.step.embedding .title']);
		},
		out: () => {
			removeHighlightFromElements(['.step.embedding .title']);
		},
		complete: () => {
			removeFingerFromElements(['.step.embedding .title']);
			if (get(textbookCurrentPageId) === 'embedding') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'embedding'
				});
			}
		}
	},
	{
		id: 'token-embedding',
		title: 'Token 嵌入',
		content: `<p><strong>分词</strong>将输入文本分割成 token——像单词或单词的一部分这样的小单元。GPT-2 (small) 有 50,257 个 token 词汇表,每个都有唯一的 ID。</p><p>在 <strong>token 嵌入</strong>步骤中,每个 token 都从一个大型查找表中匹配到一个 768 个数字的向量。这些向量在训练期间学习,以最好地表示每个 token 的含义。</p>`,
		on: function () {
			const selectors = [
				'.token-column .column.token-string',
				'.token-column .column.token-embedding'
			];
			if (get(expandedBlock).id !== 'embedding') {
				expandedBlock.set({ id: 'embedding' });
				this.timeoutId = setTimeout(() => {
					highlightElements(selectors);
				}, 500);
			} else {
				highlightElements(selectors);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			const selectors = [
				'.token-column .column.token-string',
				'.token-column .column.token-embedding'
			];
			removeHighlightFromElements(selectors);
			if (get(textbookCurrentPageId) !== 'positional-encoding') expandedBlock.set({ id: null });
		}
	},
	{
		id: 'positional-encoding',
		title: '位置编码',
		content: `<p>语言中的词序很重要。<strong>位置编码</strong>为每个 token 提供关于其在序列中位置的信息。</p><p>GPT-2 通过将学习到的位置嵌入添加到 token 的嵌入中来实现这一点,但较新的模型可能使用其他方法,如 RoPE,它通过旋转某些向量来编码位置。所有方法都旨在帮助模型理解文本中的顺序。</p>`,
		on: function () {
			const selectors = [
				'.token-column .column.position-embedding',
				'.token-column .column.symbol'
			];
			if (get(expandedBlock).id !== 'embedding') {
				expandedBlock.set({ id: 'embedding' });
				this.timeoutId = setTimeout(() => {
					highlightElements(selectors);
				}, 500);
			} else {
				highlightElements(selectors);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			const selectors = [
				'.token-column .column.position-embedding',
				'.token-column .column.symbol'
			];
			removeHighlightFromElements(selectors);
			if (get(textbookCurrentPageId) !== 'token-embedding') expandedBlock.set({ id: null });
		}
	},
	{
		id: 'blocks',
		title: '重复的 Transformer 块',
		content: `<p><strong>Transformer 块</strong>是模型中的主要处理单元。它有两个部分:</p><ul><li><strong>多头自注意力</strong> – 让 token 共享信息</li><li><strong>MLP</strong> – 细化每个 token 的细节</li></ul><p>模型堆叠许多块,因此 token 表示在通过时变得更加丰富。GPT-2 (small) 有 12 个这样的块。</p>`,
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements([
						'.transformer-bounding',
						'.step.transformer-blocks .guide',
						'.attention > .title',
						'.mlp > .title'
					]);
					highlightElements(['.transformer-bounding-title'], 'textbook-button-highlight');
					isBoundingBoxActive.set(true);
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements([
				'.transformer-bounding',
				'.step.transformer-blocks .guide',
				'.attention > .title',
				'.mlp > .title'
			]);
			removeHighlightFromElements(['.transformer-bounding-title'], 'textbook-button-highlight');
			isBoundingBoxActive.set(false);
		},
		complete: () => {
			removeFingerFromElements(['.transformer-bounding-title']);
			if (get(textbookCurrentPageId) === 'blocks') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'blocks'
				});
			}
		}
	},
	{
		id: 'self-attention',
		title: '多头自注意力',
		content:
			'<p><strong>自注意力</strong>让模型决定输入的哪些部分与每个 token 最相关。这有助于它捕获含义和关系,即使是相距很远的单词之间。</p><p>在<strong>多头</strong>形式中,模型并行运行多个注意力过程,每个都关注文本中的不同模式。</p>',
		on: () => {
			highlightElements(['.step.attention']);
		},
		out: () => {
			removeHighlightFromElements(['.step.attention']);
		}
	},
	{
		id: 'qkv',
		title: 'Query、Key、Value',
		content: `
	<p>为了执行自注意力,每个 token 的嵌入被转换为
  <span class="highlight">三个新的嵌入</span>——
  <span class="blue">Query</span>、
  <span class="red">Key</span> 和
  <span class="green">Value</span>。
  这种转换是通过对每个 token 嵌入应用不同的权重和偏置来完成的。这些参数(权重和偏置)通过训练进行优化。</p>

<p>一旦创建,<span class="blue">Query</span> 与 <span class="red">Key</span> 进行比较以衡量相关性,这种相关性用于加权 <span class="green">Value</span>。</p>
`,
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements(['g.path-group.qkv', '.step.qkv .qkv-column']);
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements(['g.path-group.qkv', '.step.qkv .qkv-column']);
			weightPopover.set(null);
		},
		complete: () => {
			removeFingerFromElements(['.step.qkv .qkv-column']);
			if (get(textbookCurrentPageId) === 'qkv') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'qkv'
				});
			}
		}
	},

	{
		id: 'multi-head',
		title: '多头',
		content:
			'<p>在创建 <span class="blue">Q</span>、<span class="red">K</span> 和 <span class="green">V</span> 嵌入后,模型将它们分成几个<strong>头</strong>(GPT-2 small 中有 12 个)。每个头使用自己的较小的 <span class="blue">Q</span>/<span class="red">K</span>/<span class="green">V</span> 集合,关注文本中的不同模式——如语法、含义或长距离链接。</p><p>多个头让模型并行学习多种关系,使其理解更加丰富。</p>',
		on: () => {
			highlightAttentionPath();
			highlightElements(['.multi-head .head-title']);
		},
		out: () => {
			removeAttentionPathHighlight();
			removeHighlightFromElements(['.multi-head .head-title']);
		},
		complete: () => {
			removeFingerFromElements(['.multi-head .head-title']);
			if (get(textbookCurrentPageId) === 'multi-head') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'multi-head'
				});
			}
		}
	},
	{
		id: 'masked-self-attention',
		title: '掩码自注意力',
		content: `<p>在每个头中,模型决定每个 token 对其他 token 的关注程度:</p><ul><li><strong>点积</strong> – 将 <span class="blue">Query</span>/<span class="red">Key</span> 向量中的匹配数字相乘,求和得到 <span class="purple">注意力分数</span>。</li><li><strong>掩码</strong> – 隐藏未来的 token,使其无法提前窥视。</li><li><strong>Softmax</strong> – 将分数转换为概率,每行总和为 1,显示对早期 token 的关注。</li></ul>`,
		on: () => {
			highlightAttentionPath();
			highlightElements(['.attention-matrix.attention-result']);
		},
		out: () => {
			removeAttentionPathHighlight();
			removeHighlightFromElements(['.attention-matrix.attention-result']);
			expandedBlock.set({ id: null });
		},
		complete: () => {
			removeFingerFromElements(['.attention-matrix.attention-result']);
			if (get(textbookCurrentPageId) === 'masked-self-attention') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'masked-self-attention'
				});
			}
		}
	},
	{
		id: 'output-concatenation',
		title: '注意力输出与拼接',
		content:
			'<p>每个头<span class="highlight">将其 <span class="purple">注意力分数</span>与 <span class="green">Value</span> 嵌入相乘以产生其注意力输出</span>——在考虑上下文后每个 token 的精炼表示。</p><p>GPT-2 (small) 有 12 个这样的输出,它们被拼接形成原始大小的单个向量(768 个数字)。</p>',
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements(['path.to-attention-out.value-to-out', '.attention .column.out']);
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements(['path.to-attention-out.value-to-out', '.attention .column.out']);
			weightPopover.set(null);
		},
		complete: () => {
			removeFingerFromElements(['.attention .column.out']);
			if (get(textbookCurrentPageId) === 'output-concatenation') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'output-concatenation'
				});
			}
		}
	},
	{
		id: 'mlp',
		title: 'MLP (多层感知器)',
		content:
			'<p>注意力输出通过 <strong>MLP</strong> 来细化 token 表示。线性层使用学习到的权重和偏置改变嵌入值和大小,然后非线性激活决定每个值通过多少。</p><p>存在许多激活类型;GPT-2 使用 <strong>GELU</strong>,它让小值部分通过,大值完全通过,有助于捕获微妙和强烈的模式。</p>',
		on: () => {
			highlightElements(['.step.mlp', '.operation-col.activation']);
		},
		out: () => {
			removeHighlightFromElements(['.step.mlp', '.operation-col.activation']);
		}
	},

	{
		id: 'output-logit',
		title: '输出 Logit',
		content: `<p>在所有 Transformer 块之后,最后一个 token 的输出嵌入(富含所有先前 token 的上下文)在最后一层中与学习到的权重相乘。</p><p>这产生 <strong>logits</strong>,50,257 个数字——GPT-2 词汇表中每个 token 一个——表示每个 token 接下来出现的可能性。</p>`,
		on: () => {
			highlightElements(['g.path-group.softmax', '.column.final']);
		},
		out: () => {
			removeHighlightFromElements(['g.path-group.softmax', '.column.final']);
			weightPopover.set(null);
		},
		complete: () => {
			removeFingerFromElements(['.column.final']);
			if (get(textbookCurrentPageId) === 'output-logit') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'output-logit'
				});
			}
		}
	},
	{
		id: 'output-probabilities',
		title: '概率',
		content:
			'<p>Logits 只是原始分数。为了使它们更容易解释,我们将它们转换为 0 到 1 之间的<strong>概率</strong>,所有概率加起来为 1。这告诉我们每个 token 成为下一个词的可能性。</p><p>我们可以使用不同的选择策略来平衡生成文本的安全性和创造性,而不是总是选择概率最高的 token。</p>',
		on: () => {
			highlightElements(['.step.softmax .title']);
		},
		out: () => {
			removeHighlightFromElements(['.step.softmax .title']);
		},
		complete: () => {
			removeFingerFromElements(['.step.softmax .title']);
			if (get(textbookCurrentPageId) === 'output-probabilities') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'output-probabilities'
				});
			}
		}
	},
	{
		id: 'temperature',
		title: '温度',
		content:
			'<p><strong>温度</strong>通过在将 logits 转换为概率之前对其进行缩放来工作。<strong>低温度</strong>(例如 0.2)使大的 logits 更大,小的更小,偏向得分最高的 token,导致更<strong>可预测的选择</strong>。<strong>高温度</strong>(例如 1.0 或更高)使差异变平,使不太可能的 token 更具竞争力,导致更<strong>有创意的输出</strong>。</p>',
		on: function () {
			if (get(expandedBlock).id !== 'softmax') {
				expandedBlock.set({ id: 'softmax' });
				this.timeoutId = setTimeout(() => {
					highlightElements([
						'.formula-step.scaled',
						'.title-box.scaled',
						'.content-box.scaled',
						'.temperature-input'
					]);
				}, 500);
			} else {
				highlightElements([
					'.formula-step.scaled',
					'.title-box.scaled',
					'.content-box.scaled',
					'.temperature-input'
				]);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements([
				'.formula-step.scaled',
				'.title-box.scaled',
				'.temperature-input',
				'.content-box.scaled'
			]);
			if (!['temperature', 'sampling'].includes(get(textbookCurrentPageId)))
				expandedBlock.set({ id: null });
		},
		complete: () => {
			removeFingerFromElements(['.temperature-input']);
			if (get(textbookCurrentPageId) === 'temperature') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'temperature'
				});
			}
		}
	},
	{
		id: 'sampling',
		title: '采样策略',
		content:
			'<p>最后,我们需要一个策略来选择下一个 token。存在许多策略,但这里是常见的:贪婪搜索选择最高的一个。<strong>Top-k</strong> 只保留 k 个最可能的 token,<strong>top-p</strong> 保留总概率至少为 p 的最小集合——提前修剪不太可能的 token。</p><p>然后 softmax 将剩余的 logits 转换为概率,并从允许的集合中随机选择一个 token。</p>',
		on: function () {
			if (get(expandedBlock).id !== 'softmax') {
				expandedBlock.set({ id: 'softmax' });
				this.timeoutId = setTimeout(() => {
					highlightElements([
						'.formula-step.sampling',
						'.title-box.sampling',
						'.sampling-input',
						'.content-box.sampling'
					]);
				}, 500);
			} else {
				highlightElements([
					'.formula-step.sampling',
					'.title-box.sampling',
					'.sampling-input',
					'.content-box.sampling'
				]);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements([
				'.formula-step.sampling',
				'.title-box.sampling',
				'.sampling-input',
				'.content-box.sampling'
			]);
			if (!['temperature', 'sampling'].includes(get(textbookCurrentPageId)))
				expandedBlock.set({ id: null });
		},
		complete: () => {
			removeFingerFromElements(['.sampling-input']);
			if (get(textbookCurrentPageId) === 'sampling') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'sampling'
				});
			}
		}
	},
	{
		id: 'residual',
		title: '残差连接',
		content: `<p>Transformer 具有增强模型性能的辅助功能。例如,<strong>残差连接</strong>将层的输入添加到其输出,防止信息在许多块中消失。在 GPT-2 中,每个块使用两次以有效训练更深的堆栈。</p>`,
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements(['.operation-col.residual', '.residual-start']);
					drawLine();
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements(['.operation-col.residual', '.residual-start']);
			removeLine();
		}
	},
	{
		id: 'layer-normalization',
		title: '层归一化',
		content: `<p><strong>层归一化</strong>通过调整输入数字使其均值和方差保持一致,有助于稳定训练和推理。这使模型对其初始权重不那么敏感,并帮助它更有效地学习。在 GPT-2 中,它在自注意力之前、MLP 之前以及最终输出之前再次应用。</p>`,
		on: () => {
			highlightElements(['.operation-col.ln']);
		},
		out: () => {
			removeHighlightFromElements(['.operation-col.ln']);
		}
	},
	{
		id: 'dropout',
		title: 'Dropout',
		content: `<p>在训练期间,<strong>dropout</strong> 随机关闭数字之间的一些连接,使模型不会过度拟合特定模式。这有助于它学习更好泛化的特征。GPT-2 使用它,但较新的 LLM 通常跳过它,因为它们在巨大的数据集上训练,过拟合不太成问题。在推理中,dropout 被关闭。</p>`,
		on: () => {
			highlightElements(['.operation-col.dropout']);
		},
		out: () => {
			removeHighlightFromElements(['.operation-col.dropout']);
		}
	}
	// {
	// 	id: 'final',
	// 	title: `Let's explore!`,
	// 	content: '',
	// 	on: () => {},
	// 	out: () => {}
	// }
];
