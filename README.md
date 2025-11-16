<div align="center">
    <h1 align="center">
        <img src="./assets/logo.png" alt="logo" height="40" style="vertical-align: middle; margin-right: 5px;" />
        <b><em>MetaGPT: A Large Vision-Language Model for Meme Metaphor Understanding</em></b>
    </h1>
    <a href="https://meizhiyuan88666.github.io/MetaGPT.github.io/" target="_blank"  style="display:inline-block;">
        <img alt="Website" src="https://img.shields.io/badge/🌎Website-Meta--GPT-blue.svg" height="25"/>
    </a>
    <a>
        <img src="https://img.shields.io/github/stars/MeiZhiyuan88666/MetaGPT?style=flat&logo=github" alt="GitHub stars" height="25"/>
    </a>
    <a>
        <img src="https://img.shields.io/github/forks/MeiZhiyuan88666/MetaGPT?style=flat&logo=github" alt="GitHub forks" height="25"/>
    </a>
    <br><br>
    <a href="https://huggingface.co/collections/MM-ZY/mund-datasets" target="_blank">
    	<img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface" alt="HuggingFace Dataset" height="25">
	</a>
    <h2 align="center"><em>Accepted By AAAI 2026!</em></h2>
</div>
  <p>
      🌈 We introduce <strong>MetaGPT</strong>, the first vision-language model specifically designed for meme metaphor understanding. MetaGPT is capable of identifying and extracting metaphors in memes, and generating accurate meme interpretations. Furthermore, we construct a dedicated dataset for meme understanding, <strong>MUnd</strong>. Based on MUnd, we further propose an evaluation benchmark for meme understanding and conduct a comprehensive assessment of existing VLMs. Experimental results reveal that current models still face challenges in metaphor comprehension, while MetaGPT consistently outperforms them across all tasks, highlighting its potential in advancing meme understanding. Our code and appendix are available in the supplementary materials.
  </p>


## 🖼️ Datasets

The image sources are available in the following referenced works.

🥽 [**MET-Meme**](https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors)

🥽 [**MEMECAP**](https://github.com/eujhwang/meme-cap)

The MUnd dataset and the benchmark test set can be accessed via the links below.

🤗 <a href="https://huggingface.co/collections/MM-ZY/mund-datasets">**Datasets and Benchmark**</a>

### Example

<div align="center">
  <a href="">
    <img src="assets/F7-2.png" alt="Logo" style="width: 65%;">
  </a>
</div>


## 🤖 Model

we build our project based on the [LLaVA](https://github.com/haotian-liu/LLaVA) codebase, using "LLaVA-v1.5-7B" as the backbone model. After training, we obtain MetaGPT.

## 💯 Evaluation

For evaluation, we provide the code for the metaphor domain extraction task under `eval/META_EXTRACT.py`.

Your prediction results must follow the same format as the ground truth.

Our evaluation prompt is shown below.

<div align="center">
  <a href="">
    <img src="assets/F9.png" alt="Logo" style="width: 60%;">
  </a>
</div>


## 🫶🏻 Acknowledgement

- [**LLava**](https://github.com/haotian-liu/LLaVA)
- [**MET-Meme**](https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors)
- [**MEMECAP**](https://github.com/eujhwang/meme-cap)
