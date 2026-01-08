<script>
	import tailwindConfig from '../../../tailwind.config';
	import resolveConfig from 'tailwindcss/resolveConfig';
	import Katex from '~/utils/Katex.svelte';
</script>

<div id="description">
	<div class="article-section" data-click="article-intro">
		<h1>什么是 Transformer?</h1>

		<p>
			Transformer 是一种神经网络架构,从根本上改变了人工智能的方法。Transformer 首次在开创性论文
			<a
				href="https://dl.acm.org/doi/10.5555/3295222.3295349"
				title="ACM Digital Library"
				target="_blank">"Attention is All You Need"</a
			>
			中于 2017 年提出,此后成为深度学习模型的首选架构,为 OpenAI 的 <strong>GPT</strong>、Meta 的 <strong>Llama</strong> 和 Google 的
			<strong>Gemini</strong> 等文本生成模型提供支持。除了文本,Transformer 还应用于
			<a
				href="https://huggingface.co/learn/audio-course/en/chapter3/introduction"
				title="Hugging Face"
				target="_blank">音频生成</a
			>、
			<a
				href="https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification"
				title="Hugging Face"
				target="_blank">图像识别</a
			>、
			<a href="https://elifesciences.org/articles/82819" title="eLife"
				>蛋白质结构预测</a
			>,甚至
			<a
				href="https://www.deeplearning.ai/the-batch/reinforcement-learning-plus-transformers-equals-efficiency/"
				title="Deep Learning AI"
				target="_blank">游戏</a
			>,展示了其在众多领域的多功能性。
		</p>
		<p>
			从根本上说,文本生成 Transformer 模型基于<strong
				>下一个 token 预测</strong
			>的原理运行:给定用户的文本提示,<em>最可能跟随此输入的下一个 token(一个词或词的一部分)</em>是什么?Transformer 的核心创新和力量在于它们使用自注意力机制,这使它们能够处理整个序列并比以前的架构更有效地捕获长距离依赖关系。
		</p>
		<p>
			GPT-2 系列模型是文本生成 Transformer 的突出例子。Transformer
			Explainer 由
			<a href="https://huggingface.co/openai-community/gpt2" title="Hugging Face" target="_blank"
				>GPT-2</a
			>
			(small) 模型提供支持,该模型有 1.24 亿个参数。虽然它不是最新或最强大的 Transformer
			模型,但它与当前最先进模型中的许多相同架构组件和原理相同,使其成为理解基础知识的理想起点。
		</p>
	</div>

	<div class="article-section" data-click="article-overview">
		<h1>Transformer 架构</h1>

		<p>
			每个文本生成 Transformer 都由这<strong>三个关键组件</strong>组成:
		</p>
		<ol>
			<li>
				<strong class="bold-purple">嵌入</strong>:文本输入被分成称为 token 的较小单元,可以是单词或子词。这些 token 被转换为称为嵌入的数值向量,捕获单词的语义含义。
			</li>
			<li>
				<strong class="bold-purple">Transformer 块</strong>是处理和转换输入数据的模型的基本构建块。每个块包括:
				<ul class="">
					<li>
						<strong>注意力机制</strong>,Transformer 块的核心组件。它允许 token 与其他 token 通信,捕获上下文信息和单词之间的关系。
					</li>
					<li>
						<strong>MLP(多层感知器)层</strong>,一个独立操作每个 token 的前馈网络。注意力层的目标是在 token 之间路由信息,而 MLP 的目标是细化每个 token 的表示。
					</li>
				</ul>
			</li>
			<li>
				<strong class="bold-purple">输出概率</strong>:最终的线性层和 softmax 层将处理后的嵌入转换为概率,使模型能够对序列中的下一个 token 进行预测。
			</li>
		</ol>
	</div>

	<div class="article-section" id="embedding" data-click="article-embedding">
		<h2>嵌入</h2>
		<p>
			假设你想使用 Transformer 模型生成文本。你添加这样的提示:<code>"Data visualization empowers users to"</code>。这个输入需要转换为模型可以理解和处理的格式。这就是嵌入的作用:它将文本转换为模型可以使用的数值表示。要将提示转换为嵌入,我们需要 1) 对输入进行分词,2) 获取 token 嵌入,3) 添加位置信息,最后 4) 将 token 和位置编码相加以获得最终嵌入。让我们看看如何完成这些步骤。
		</p>
		<div class="figure">
			<img src="./article_assets/embedding.png" width="65%" />
		</div>
		<div class="figure-caption">
			图 <span class="attention">1</span>. 展开嵌入层视图,显示如何将输入提示转换为向量表示。该过程包括
			<span class="fig-numbering">(1)</span> 分词,(2) Token 嵌入,(3) 位置编码,以及 (4) 最终嵌入。
		</div>
		<div class="article-subsection">
			<h3>步骤 1:分词</h3>
			<p>
				分词是将输入文本分解为称为 token 的更小、更易管理的片段的过程。这些 token 可以是一个词或一个子词。单词 <code>"Data"</code>
				和 <code>"visualization"</code> 对应于唯一的 token,而单词
				<code>"empowers"</code>
				被分成两个 token。token 的完整词汇表在训练模型之前确定:
				GPT-2 的词汇表有 <code>50,257</code> 个唯一 token。现在我们将输入文本分成具有不同 ID 的 token,我们可以从嵌入中获得它们的向量表示。
			</p>
		</div>
		<div class="article-subsection" id="article-token-embedding">
			<h3>步骤 2. Token 嵌入</h3>
			<p>
				GPT-2 (small) 将词汇表中的每个 token 表示为 768 维向量;向量的维度取决于模型。这些嵌入向量存储在形状为 <code>(50,257, 768)</code> 的矩阵中,包含大约 3900 万个参数!这个广泛的矩阵允许模型为每个 token 分配语义含义,即在语言中具有相似用法或含义的 token 在这个高维空间中被放置得很近,而不相似的 token 则相距较远。
			</p>
		</div>
		<div class="article-subsection" id="article-positional-embedding">
			<h3>步骤 3. 位置编码</h3>
			<p>
				嵌入层还编码有关每个 token 在输入提示中位置的信息。不同的模型使用各种位置编码方法。GPT-2 从头开始训练自己的位置编码矩阵,将其直接集成到训练过程中。
			</p>

			<!-- <div class="article-subsection-l2">
	<h4>Alternative Positional Encoding Approach <strong class='attention'>[POTENTIALLY COLLAPSIBLE]</strong></h4>
	<p>
	  Other models, like the original Transformer and BERT,
	  use sinusoidal functions for positional encoding.

	  This sinusoidal encoding is deterministic and designed to reflect
	  the absolute as well as the relative position of each token.
	</p>
	<p>
	  Each position in a sequence is assigned a unique mathematical
	  representation using a combination of sine and cosine functions.

	  For a given position, the sine function represents even dimensions,
	  and the cosine function represents odd dimensions within the positional encoding vector.

	  This periodic nature ensures that each position has a consistent encoding,
	  independent of the surrounding context.
	</p>

	<p>
	  Here's how it works:
	</p>

	<span class='attention'>
	  SINUSOIDAL POSITIONAL ENCODING EQUATION
	</span>

	<ul>
	  <li>
		<strong>Sine Function</strong>: Used for even indices of the embedding vector.
	  </li>
	  <li>
		<strong>Cosine Function</strong>: Used for odd indices of the embedding vector.
	</ul>

	<p>
	  Hover over individual encoding values in the matrix above to
	  see how it's calculated using the sins and cosine functions.
	</p>
  </div> -->
		</div>
		<div class="article-subsection">
			<h3>步骤 4. 最终嵌入</h3>
			<p>
				最后,我们将 token 和位置编码相加以获得最终的嵌入表示。这种组合表示既捕获了 token 的语义含义,又捕获了它们在输入序列中的位置。
			</p>
		</div>
	</div>

	<div class="article-section" data-click="article-transformer-block">
		<h2>Transformer 块</h2>

		<p>
			Transformer 处理的核心在于 Transformer 块,它包括多头自注意力和多层感知器层。大多数模型由多个这样的块组成,这些块按顺序依次堆叠。token 表示通过层演化,从第一个块到最后一个块,允许模型建立对每个 token 的复杂理解。这种分层方法导致输入的高阶表示。我们正在检查的 GPT-2 (small) 模型由 <code
				>12</code
			> 个这样的块组成。
		</p>
	</div>

	<div class="article-section" id="self-attention" data-click="article-attention">
		<h3>多头自注意力</h3>
		<p>
			自注意力机制使模型能够捕获序列中 token 之间的关系,以便每个 token 的表示受到其他 token 的影响。多个注意力头允许模型从不同的角度考虑这些关系;例如,一个头可能捕获短距离句法链接,而另一个头跟踪更广泛的语义上下文。在下一节中,我们将逐步介绍如何计算多头自注意力。
		</p>
		<div class="article-subsection-l2">
			<h4>步骤 1:Query、Key 和 Value 矩阵</h4>

			<div class="figure pt-10">
				<img src="./article_assets/QKV.png" width="80%" />
				<div class="text-xs">
					<Katex
						displayMode
						math={`
		QKV_{ij} = ( \\sum_{d=1}^{768} \\text{Embedding}_{i,d} \\cdot \\text{Weights}_{d,j}) + \\text{Bias}_j
		`}
					/>
				</div>
			</div>
			<div class="figure-caption">
				图 <span class="attention">2</span>. 从原始嵌入计算 Query、Key 和 Value 矩阵。
			</div>

			<p>
				每个 token 的嵌入向量被转换为三个向量:
				<span class="q-color">Query (Q)</span>、
				<span class="k-color">Key (K)</span> 和
				<span class="v-color">Value (V)</span>。这些向量是通过将输入嵌入矩阵与学习到的权重矩阵相乘得出的,用于
				<span class="q-color">Q</span>、
				<span class="k-color">K</span> 和
				<span class="v-color">V</span>。这里有一个网络搜索类比来帮助我们建立对这些矩阵的直觉:
			</p>
			<ul>
				<li>
					<strong class="q-color font-medium">Query (Q)</strong> 是你在搜索引擎栏中输入的搜索文本。这是你想要<em>"查找更多信息"</em>的 token。
				</li>
				<li>
					<strong class="k-color font-medium">Key (K)</strong> 是搜索结果窗口中每个网页的标题。它表示查询可以关注的可能 token。
				</li>
				<li>
					<strong class="v-color font-medium">Value (V)</strong> 是显示的网页的实际内容。一旦我们将适当的搜索词(Query)与相关结果(Key)匹配,我们就想获得最相关页面的内容(Value)。
				</li>
			</ul>
			<p>
				通过使用这些 QKV 值,模型可以计算注意力分数,这些分数确定在生成预测时每个 token 应该获得多少关注。
			</p>
		</div>
		<div class="article-subsection-l2">
			<h4>步骤 2:多头分割</h4>
			<p>
				<span class="q-color">Query</span>、<span class="k-color">key</span> 和
				<span class="v-color">Value</span>
				向量被分成多个头——在 GPT-2 (small) 的情况下,分成
				<code>12</code> 个头。每个头独立处理嵌入的一个片段,捕获不同的句法和语义关系。这种设计促进了不同语言特征的并行学习,增强了模型的表示能力。
			</p>
		</div>
		<div class="article-subsection-l2">
			<h4>步骤 3:掩码自注意力</h4>
			<p>
				在每个头中,我们执行掩码自注意力计算。这种机制允许模型通过关注输入的相关部分来生成序列,同时防止访问未来的 token。
			</p>

			<div class="figure">
				<img src="./article_assets/attention.png" width="80%" align="middle" />
			</div>
			<div class="figure-caption">
				图 <span class="attention">3</span>. 使用 Query、Key 和 Value 矩阵计算掩码自注意力。
			</div>

			<ul>
				<li>
					<strong>点积</strong>:<span class="q-color">Query</span>
					和 <span class="k-color">Key</span> 矩阵的点积确定
					<strong>注意力分数</strong>,产生一个反映所有输入 token 之间关系的方阵。
				</li>
				<li>
					<strong>缩放 · 掩码</strong>:注意力分数被缩放,并对注意力矩阵的上三角应用掩码以防止模型访问未来的 token,将这些值设置为负无穷大。模型需要学习如何在不"窥视"未来的情况下预测下一个 token。
				</li>
				<li>
					<strong>Softmax · Dropout</strong>:在掩码和缩放之后,注意力分数通过 softmax 操作转换为概率,然后可选地使用 dropout 进行正则化。矩阵的每一行总和为 1,并指示其左侧每个其他 token 的相关性。
				</li>
			</ul>
		</div>
		<div class="article-subsection-l2">
			<h4>步骤 4:输出和拼接</h4>
			<p>
				模型使用掩码自注意力分数并将它们与
				<span class="v-color">Value</span> 矩阵相乘以获得自注意力机制的
				<span class="purple-color">最终输出</span>。GPT-2 有 <code>12</code> 个自注意力头,每个头捕获 token 之间的不同关系。这些头的输出被拼接并通过线性投影传递。
			</p>
		</div>
	</div>

	<div class="article-section" id="article-activation" data-click="article-mlp">
		<h3>MLP:多层感知器</h3>

		<div class="figure">
			<img src="./article_assets/mlp.png" width="70%" align="middle" />
		</div>
		<div class="figure-caption">
			图 <span class="attention">4</span>. 使用 MLP 层将自注意力表示投影到更高维度以增强模型的表示能力。
		</div>

		<p>
			在多个自注意力头捕获输入 token 之间的多样关系之后,拼接的输出通过多层感知器(MLP)层以增强模型的表示能力。MLP 块由两个线性变换组成,中间有一个 <a
				href="https://en.wikipedia.org/wiki/Rectified_linear_unit#Gaussian-error_linear_unit_(GELU)"
				>GELU</a
			> 激活函数。
		</p>
		<p>
			第一个线性变换将输入的维度从 <code
				>768</code
			>
			扩展四倍到
			<code>3072</code>。这个扩展步骤允许模型将 token 表示投影到更高维空间,在那里它可以捕获在原始维度中可能不可见的更丰富和更复杂的模式。
		</p>
		<p>
			然后第二个线性变换将维度减少回原始大小 <code
				>768</code
			>。这个压缩步骤将表示带回可管理的大小,同时保留在扩展步骤中引入的有用非线性变换。
		</p>
		<p>
			与跨 token 集成信息的自注意力机制不同,MLP 独立处理 token,只是将每个 token 表示从一个空间映射到另一个空间,丰富了整体模型容量。
		</p>
	</div>

	<div class="article-section" id="article-prob" data-click="article-prob">
		<h2>输出概率</h2>
		<p>
			在输入通过所有 Transformer 块处理之后,输出通过最终线性层以准备进行 token 预测。该层将最终表示投影到 <code>50,257</code>
			维空间,其中词汇表中的每个 token 都有一个称为
			<code>logit</code> 的对应值。任何 token 都可以是下一个词,因此这个过程允许我们简单地按它们成为下一个词的可能性对这些 token 进行排名。然后我们应用 softmax 函数将 logits 转换为总和为 1 的概率分布。这将允许我们根据其可能性对下一个 token 进行采样。
		</p>

		<div class="figure py-5">
			<img src="./article_assets/softmax.png" width="70%" />
		</div>
		<div class="figure-caption">
			图 <span class="attention">5</span>. 词汇表中的每个 token 都根据模型的输出 logits 分配一个概率。这些概率确定每个 token 成为序列中下一个词的可能性。
		</div>

		<p id="article-temperature" data-click="article-temperature">
			最后一步是通过从这个分布中采样来生成下一个 token。<code
				>temperature</code
			>
			超参数在这个过程中起着关键作用。从数学上讲,这是一个非常简单的操作:模型输出 logits 简单地除以
			<code>temperature</code>:
		</p>

		<ul>
			<li>
				<code>temperature = 1</code>:将 logits 除以 1 对 softmax 输出没有影响。
			</li>
			<li>
				<code>temperature &lt; 1</code>:较低的温度通过锐化概率分布使模型更加自信和确定性,导致更可预测的输出。
			</li>
			<li>
				<code>temperature &gt; 1</code>:较高的温度创建更柔和的概率分布,允许生成的文本中有更多的随机性——有些人称之为模型<em>"创造力"</em>。
			</li>
		</ul>

		<p id="article-sampling" data-click="article-sampling">
			此外,采样过程可以使用 <code>top-k</code>
			和
			<code>top-p</code> 参数进一步细化:
		</p>
		<ul>
			<li>
				<code>top-k 采样</code>:将候选 token 限制为概率最高的前 k 个 token,过滤掉不太可能的选项。
			</li>
			<li>
				<code>top-p 采样</code>:考虑累积概率超过阈值 p 的最小 token 集,确保只有最可能的 token 贡献,同时仍然允许多样性。
			</li>
		</ul>
		<p>
			通过调整 <code>temperature</code>、<code>top-k</code> 和 <code>top-p</code>,你可以在确定性和多样性输出之间取得平衡,根据你的特定需求定制模型的行为。
		</p>
	</div>

	<div class="article-section" data-click="article-advanced-features">
		<h2>辅助架构特征</h2>

		<p>
			有几个辅助架构特征可以增强 Transformer 模型的性能。虽然对模型的整体性能很重要,但它们对于理解架构的核心概念并不那么重要。层归一化、Dropout 和残差连接是 Transformer 模型中的关键组件,特别是在训练阶段。层归一化稳定训练并帮助模型更快收敛。Dropout 通过随机停用神经元来防止过拟合。残差连接允许梯度直接流过网络,有助于防止梯度消失问题。
		</p>
		<div class="article-subsection" id="article-ln">
			<h3>层归一化</h3>

			<p>
				层归一化有助于稳定训练过程并改善收敛。它通过跨特征归一化输入来工作,确保激活的均值和方差是一致的。这种归一化有助于缓解与内部协变量偏移相关的问题,允许模型更有效地学习并降低对初始权重的敏感性。层归一化在每个 Transformer 块中应用两次,一次在自注意力机制之前,一次在 MLP 层之前。
			</p>
		</div>
		<div class="article-subsection" id="article-dropout">
			<h3>Dropout</h3>

			<p>
				Dropout 是一种正则化技术,用于通过在训练期间随机将一部分模型权重设置为零来防止神经网络中的过拟合。这鼓励模型学习更健壮的特征并减少对特定神经元的依赖,帮助网络更好地泛化到新的、未见过的数据。在模型推理期间,dropout 被停用。这本质上意味着我们正在使用训练子网络的集成,这导致更好的模型性能。
			</p>
		</div>
		<div class="article-subsection" id="article-residual">
			<h3>残差连接</h3>

			<p>
				残差连接首次在 2015 年的 ResNet 模型中引入。这种架构创新通过使训练非常深的神经网络成为可能而彻底改变了深度学习。本质上,残差连接是绕过一个或多个层的快捷方式,将层的输入添加到其输出。这有助于缓解梯度消失问题,使训练堆叠在一起的多个 Transformer 块的深度网络变得更容易。在 GPT-2 中,残差连接在每个 Transformer 块内使用两次:一次在 MLP 之前,一次在之后,确保梯度更容易流动,并且早期层在反向传播期间接收足够的更新。
			</p>
		</div>
	</div>

	<div class="article-section" data-click="article-interactive-features">
		<h1>交互功能</h1>
		<p>
			Transformer Explainer 构建为交互式的,允许你探索 Transformer 的内部工作原理。以下是你可以使用的一些交互功能:
		</p>

		<ul>
			<li>
				<strong>输入你自己的文本序列</strong>以查看模型如何处理它并预测下一个词。探索注意力权重、中间计算,并查看如何计算最终输出概率。
			</li>
			<li>
				<strong>使用温度滑块</strong>来控制模型预测的随机性。探索如何通过更改温度值使模型输出更确定性或更有创意。
			</li>
			<li>
				<strong>选择 top-k 和 top-p 采样方法</strong>以在推理期间调整采样行为。尝试不同的值,看看概率分布如何变化并影响模型的预测。
			</li>
			<li>
				<strong>与注意力图交互</strong>以查看模型如何关注输入序列中的不同 token。将鼠标悬停在 token 上以突出显示它们的注意力权重,并探索模型如何捕获上下文和单词之间的关系。
			</li>
		</ul>
	</div>

	<div class="article-section" data-click="article-video">
		<h2>视频教程</h2>
		<div class="video-container">
			<iframe
				src="https://www.youtube.com/embed/ECR4oAwocjs"
				frameborder="0"
				allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
				allowfullscreen
			>
			</iframe>
		</div>
	</div>

	<div class="article-section" data-click="article-implementation">
		<h2>Transformer Explainer 是如何实现的?</h2>
		<p>
			Transformer Explainer 具有直接在浏览器中运行的实时 GPT-2 (small) 模型。该模型源自 Andrej Karpathy 的
			<a href="https://github.com/karpathy/nanoGPT" title="Github" target="_blank"
				>nanoGPT 项目</a
			>
			的 GPT PyTorch 实现,并已转换为
			<a href="https://onnxruntime.ai/" title="ONNX" target="_blank">ONNX Runtime</a>
			以实现无缝的浏览器内执行。界面使用 JavaScript 构建,使用
			<a href="https://kit.svelte.dev/" title="Svelte" target="_blank">Svelte</a>
			作为前端框架,使用
			<a href="https://d3js.org/" title="D3" target="_blank">D3.js</a>
			创建动态可视化。数值在用户输入后实时更新。
		</p>
	</div>

	<div class="article-section" data-click="article-credit">
		<h2>谁开发了 Transformer Explainer?</h2>
		<p>
			Transformer Explainer 由

			<a href="https://aereeeee.github.io/" target="_blank">Aeree Cho</a>、
			<a href="https://www.linkedin.com/in/chaeyeonggracekim/" target="_blank">Grace C. Kim</a>、
			<a href="https://alexkarpekov.com/" target="_blank">Alexander Karpekov</a>、
			<a href="https://alechelbling.com/" target="_blank">Alec Helbling</a>、
			<a href="https://zijie.wang/" target="_blank">Jay Wang</a>、
			<a href="https://seongmin.xyz/" target="_blank">Seongmin Lee</a>、
			<a href="https://bhoov.com/" target="_blank">Benjamin Hoover</a> 和
			<a href="https://poloclub.github.io/polochau/" target="_blank">Polo Chau</a>

			在佐治亚理工学院创建。
		</p>
	</div>
</div>

<style lang="scss">
	a {
		color: theme('colors.blue.500');

		&:hover {
			color: theme('colors.blue.700');
		}
	}

	.bold-purple {
		color: theme('colors.purple.700');
		font-weight: bold;
	}

	code {
		color: theme('colors.gray.500');
		background-color: theme('colors.gray.50');
		font-family: theme('fontFamily.mono');
	}

	.q-color {
		color: theme('colors.blue.400');
	}

	.k-color {
		color: theme('colors.red.400');
	}

	.v-color {
		color: theme('colors.green.400');
	}

	.purple-color {
		color: theme('colors.purple.500');
	}

	.article-section {
		padding-bottom: 2rem;
	}
	.architecture-section {
		padding-top: 1rem;
	}
	.video-container {
		position: relative;
		padding-bottom: 56.25%; /* 16:9 aspect ratio */
		height: 0;
		overflow: hidden;
		max-width: 100%;
		background: #000;
	}

	.video-container iframe {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
	}

	#description {
		padding-bottom: 3rem;
		margin-left: auto;
		margin-right: auto;
		max-width: 78ch;
	}

	#description h1 {
		color: theme('colors.purple.700');
		font-size: 2.2rem;
		font-weight: 300;
		padding-top: 1rem;
	}

	#description h2 {
		// color: #444;
		color: theme('colors.purple.700');
		font-size: 2rem;
		font-weight: 300;
		padding-top: 1rem;
	}

	#description h3 {
		color: theme('colors.gray.700');
		font-size: 1.6rem;
		font-weight: 200;
		padding-top: 1rem;
	}

	#description h4 {
		color: theme('colors.gray.700');
		font-size: 1.6rem;
		font-weight: 200;
		padding-top: 1rem;
	}

	#description p {
		margin: 1rem 0;
	}

	#description p img {
		vertical-align: middle;
	}

	#description .figure-caption {
		font-size: 0.8rem;
		margin-top: 0.5rem;
		text-align: center;
		margin-bottom: 2rem;
	}

	#description ol {
		margin-left: 3rem;
		list-style-type: decimal;
	}

	#description li {
		margin: 0.6rem 0;
	}

	#description p,
	#description div,
	#description li {
		color: theme('colors.gray.600');
		line-height: 1.6;
	}

	#description small {
		font-size: 0.8rem;
	}

	#description ol li img {
		vertical-align: middle;
	}

	#description .video-link {
		color: theme('colors.blue.600');
		cursor: pointer;
		font-weight: normal;
		text-decoration: none;
	}

	#description ul {
		list-style-type: disc;
		margin-left: 2.5rem;
		margin-bottom: 1rem;
	}

	#description a:hover,
	#description .video-link:hover {
		text-decoration: underline;
	}

	.figure,
	.video {
		width: 100%;
		display: flex;
		flex-direction: column;
		align-items: center;
	}
</style>
