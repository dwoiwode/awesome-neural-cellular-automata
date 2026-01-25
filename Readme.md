# Awesome Neural Cellular Automata
<div align="center">
    A list of paper and resources regarding Neural Cellular Automata inspired by MrNerf's <a href='https://github.com/MrNeRF/awesome-3D-gaussian-splatting'>Awesome 3D Gaussian Splatting</a>.

  [**Browse the Paper List**](https://dwoiwode.github.io/awesome-neural-cellular-automata/)
</div>

> [!NOTE]
> This repository has been researched, compiled, and maintained to the best of my knowledge and ability.
> While I strive for accuracy and completeness, there may be mistakes or updates needed over time.
> I welcome suggestions, improvements, and contributions from the community.
> Please feel free to open an issue or submit a pull request if you have any recommendations or changes.
>
> To contribute, please update the `papers.yaml` and template files in `assets/`.
> You can run `build_site.py` locally to generate the `index.html` and `stats.html` respectively.
>
> Thank you for your support and collaboration!

> [!IMPORTANT]  
> This repository moved to a github.io HTML web page for a better look and more features. You can find the list here:
> [**Paper List**](https://dwoiwode.github.io/awesome-neural-cellular-automata/)


## Seminal Paper introducing Neural Cellular Automata
<table>
<tr>
<td width="150px">
<a href="https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb" target="_blank">
    <img src="assets/thumbnails/2020-02-11growingneu_mordvintsev.jpg" width="140px">
</a>
</td>
<td>

### Growing Neural Cellular Automata
Published on **2020-02-11** by

Alexander **Mordvintsev**, Ettore **Randazzo**, Eyvind **Niklasson**, Michael **Levin**

[Project Page/Paper](https://distill.pub/2020/growing-ca/) | [Code](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb)

<b>Abstract</b>
Training an end-to-end differentiable, self-organising cellular automata model of morphogenesis, able to both grow and regenerate specific patterns.

</td>
</tr>


</table>

## Implementations
- [Original reference](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb)
- [ncalib](https://github.com/dwoiwode/ncalib) - Modular Neural Cellular Automata library in PyTorch
- [CAX](https://github.com/maxencefaldor/cax) - Cellular Automata in JAX (Flax NNX)
- [JAX-NCA](https://github.com/shyamsn97/jax-nca) - NCA Implementation in JAX (Flax Linen)
- [Hexells](https://github.com/znah/hexells) - SwissGL Implementation of Hexells ([Demo](https://znah.net/hexells/))


## Further Reading
- **Videos & Tutorials**
  - [Mildly Overfitted](https://www.youtube.com/watch?v=21ACbWoF2Oo) – Tutorial & code explanation (PyTorch)
  - [Yannic Kilcher](https://www.youtube.com/watch?v=9Kec_7WFyp0) – Paper explanation

- **Courses**
  - [Artificial Life by Vassilis Papadopoulos](https://vassi.life/teaching/alife)
    - [Lecture 10: Neural Cellular Automata](https://frotaur.notion.site/Course-10-Neural-Cellular-automata-63d6eb2efe9443b4b2c3a09a55f493a0) ([Video Recording](https://www.youtube.com/watch?v=_ealiM25biA))
  - [AIAIArt Course by John Whitaker](https://github.com/johnowhitaker/aiaiart)
    - [Lesson #8: Neural CA](https://colab.research.google.com/drive/1Qpx_4wWXoiwTRTCAP1ohpoPGwDIrp9z-) ([Video Recording](https://www.youtube.com/watch?v=X2-ucB74oEk))
    - [Full YouTube Playlist](https://www.youtube.com/playlist?list=PL23FjyM69j910zCdDFVWcjSIKHbSB7NE8)

- **Projects & Tools**
  - [NeuralCA.org](https://www.neuralca.org/) | [GitHub](https://github.com/MonashDeepNeuron/Neural-Cellular-Automata)
  - [Google Self-organising Systems](https://github.com/google-research/self-organising-systems/)

- **Others**
  - Alexander Mordvintsev: [Website](https://znah.net/) | [YouTube](https://www.youtube.com/@zzznah) | [Twitter/X](https://x.com/zzznah) | [GitHub](https://github.com/znah)
  - [International Society for Artificial LIFE (ISAL)](https://alife.org/)
  - [Awesome Cellular Automata](https://github.com/vovanmozg/awesome-cellular-automata) – curated list of CA resources