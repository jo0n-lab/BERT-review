# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Keywords & Terminology

[representation](https://89douner.tistory.com/339?category=1033935); `language model`; [transformer](http://nlp.seas.harvard.edu/annotated-transformer/); `SOTA`; `GLUE`; `encoder`; `decoder`;
`sequence`; `token`; [context vector](https://wikidocs.net/24996); [word embedding](https://wikidocs.net/33520); [WordPiece](http://aidev.co.kr/nlp/7777); `BPE`; `MLM`; `NSP`;

## 초록 & 서론

BERT는 `deep` `bidirectional` `representations` 방식의 pre-train + fine tuning model

기존과는 다르게 context의 좌우를 참조하는 방식 (attend to bidirectional context)

한 개의 output layer 추가하여 여러가지 task에 맞게 학습 (task-specific)

기존의 연구들은 pre-train 모델이 다음의 tasks 에서 상당한 효용성이 있다는 것을 입증.

`natural language inference (NLI)`;  `paraphrasing`; `named entity recognition`; `question answering (QA)`;


pre-train representation 이 적용되는 downstream 두 가지 방식

두 가지 방식 모두 general language representations 학습하기 위해 같은 objective function의 단방향 모델 적용 (GPT 에서 language modeling objective 적용됬었음)—-한계

- `feature-based`
    
    task-specific architectures 가 적용되고 pre-training 마저도 추가적인 feature 의 일부(`ELMo`)
    
- `fine-tuning`
    
    pre-train 에서 얻어낸 최소한의 parameters 가 fine-tuned (기타 튜닝되듯이) (`GPT`)
    

양방향 모델을 적용하면 단방향 방식에 비해 pre-trained architecture 에 대해 더 많은 선택(왼쪽-오른쪽, 오른쪽-왼쪽)이 가능하고,  pre-train 을 fine-tuning 적용 시, 강력한 효과를 낸다. 

<aside>
👉 해당 방식은 context(문맥)을 파악하는 QA 같은 token-level tasks에서 유리해진다.

</aside>

(MLM/NSP) 적용한 pre-train model 으로 deep bidirectional transformer 을 학습한 방식 BERT 에 대해 다룬다. • `[https://github.com/google-research/bert](https://github.com/google-research/bert)` (소스코드 첨부)

## 관련하여 진행된 연구 (pre-training strategies)

**`Unsupervised Feature-based Approaches`**

 BERT와 같진 않지만 차원을 깨는(단방향) 시도로서는 ELMo를 들 수 있다. 

문맥에 민감한(cotext-sensitive) 특징들을 [왼쪽-오른쪽, 오른쪽-왼쪽]으로부터 추출한다. 추출한 representaion을 concatanate 해서 context token을 얻어낸다. 해당 방식은 major SOTA NLP model benchmarks 에서 우세한 성능을 보였다는 것에 의의가 있다. (특히 문맥을 파악하는 task에서 우세했음)

해당 방식말고도 LSTM을 활용한 not deep bidirectional 방식도 제시된 적이 있다…

**`Unsupervised Fine-tuning Approaches`**
downstream task에 적용되는 parameter을 얻기 위해 pre-training 적용한다. 해당 방식의 이점은 pre-training level에서 parameter의 학습이 진행되었기 때문에 downstream level에서는 parameter의 학습이 not-expensive하다는 것이다. 이것을 내세워 openAI GPT가 SOTA를 따냈다.

## BERT

two steps : *`pre-training` ,* *`fine-tuning`*

### Architecture

![                     그림 1 : `pre-training` , `fine-tuning` 의 architecture 가 거의 똑같이 생긴 것을 확인       ](source/Untitled.png)

                     그림 1 : `pre-training` , `fine-tuning` 의 architecture 가 거의 똑같이 생긴 것을 확인       

               

`pre-training`, `fine-tuning` 의 모양이 같은 것을 확인 (아주 작은 차이만 있다)

`fine-tuning` 의 task에 따라서도 모양이 같은 것을 확인

BERT 는 bidirectional self-attention - transformer encoder (bidirectional)

GPT 는 constrained self-attention - transformer decoder (generative)

### **Input/Output Representations**

BERT 가 임의의 문장에 대응할 수 있기 위해 token sequence 에 하나의 문장 또는 짝을 이루는 문장(2개)을 포함시켰다. (여기서 문장이란, 언어적으로 끝맺는 문장이 아닌, 연결된 text의 무작위적인 시작과 끝으로 이루어진 연결을 뜻한다.)

![                           그림 2 : sequence 의 시작을 알리는 CLS 토큰, 문장의 끝을 알리는 SEP 토큰](source/Untitled%201.png)

                           그림 2 : sequence 의 시작을 알리는 CLS 토큰, 문장의 끝을 알리는 SEP 토큰

sequence의 delimeter 로서 `CLS` 토큰은 sequence의 시작, `SEP` 토큰은 문장의 끝을 나타낸다.

또한 입력 토큰에 A 혹은 B 의 소속 정보를 추가하여 문장을 구분함.

### **Pre-training BERT [`MLM` `NSP`]**

`Masked LM` (`MLM`)

하나의 sequence 내에 무작위 토큰을 골라 *`mask`* 화 한다. (*`mask`* 화 : 원래 토큰을 다른 text로 대치)

*`mask`* 토큰이 sequence 내에 차지하는 비율을 정해서 (15%) *`mask`* 이전의 토큰을 맞추는 방식으로 학습

이렇게 하므로써 *`mask`* 토큰만 맞추면 되기 때문에 input 을 새롭게 바꿀 필요가 없음.

그러나 이는 pre-training 에만 적용되므로 downstream task 에는 *`mask`* 토큰에 대한 학습이 전혀 이뤄지지 않는다는 단점이 있다.

이를 조금이라도 해결하기 위해서 *`**mask`* 토큰 : 80%, 무작위 토큰 : 10%, 그대로 토큰 :10%** 의 확률을 부여한다. (확률적으로 부여되는 것이라 의미가 있다.)

최종적으로 $T_i$ 를 계산해서 *`mask`* 화 되기 이전의 토큰을 알아맞춘다.

`Next Sentence Prediction` (`NSP`)

language modeling objective 에서 확인하기 힘든 문맥 파악 위주의 QA, NLI

문맥 파악에 효과적인 모델을 pre-train 하기 위해 NSP 제시

monolingual corpus(하나의 언어로만 이루어진 말뭉치)로 매우 쉽게 문장을 추출하고, 이진화 시킬 수 있음(매우 간단한 방식임에 비해 QA 와 NLI에서 뛰어난 성능을 보인다.)

A, B 문장이 입력 sequence 로 주어진다면, 50%의 확률로 (옳은)B가 오고 나머지의 확률로 무작위 문장이 온다. 여기서 $C$가 문장에 대해 예측한 값이다.

![                      그림 1 : `pre-training` , `fine-tuning` 의 architecture 가 거의 똑같이 생긴 걸 확인                      ](source/Untitled.png)

                      그림 1 : `pre-training` , `fine-tuning` 의 architecture 가 거의 똑같이 생긴 걸 확인                      

`data source`

BooksCorpus (800M words), English Wikipedia (2,500M words)

**document-level corpus** >> a shuffled sentence-level corpus

### Fine-tuning **BERT**

초간단.

input, output을 갈아끼워 주기만 하면 다른 task에도 적용 가능.

실험된 tasks

`paraphrased pair` 

`hypothesis-premise pair` (이론, 가설)

`question answering pair`

`a degenerate text-∅ pair in text classification or sequence tagging`

## Experiments & Results Specs

$BERT_{BASE}$ : (L=12, H=768, A=12, Total Param- eters=110M) GPT랑 똑같은 크기

$BASE_{LARGE}$ : (L=24, H=1024, A=16, Total Parameters=340M)

### GLUE

![스크린샷 2022-07-11 오후 9.38.29.png](source/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.38.29.png)

for BERT_LARGE we found that fine- tuning was sometimes unstable on small datasets, so we ran several random restarts and selected the best model on the Dev set.

With random restarts, we use the same pre-trained checkpoint but per- form different fine-tuning data shuffling and clas- sifier layer initialization.

Note that BERTBASE and OpenAI GPT are nearly identical in terms of model architecture apart from the attention masking.

We find that BERTLARGE significantly outper- forms BERTBASE across all tasks, especially those with very little training data.

### **SQuAD v1.1, v2.0**

v1.1 은 질문과 답이 주어지는 dataset을 입력. answer 의 시작 S토큰, 끝에 E 토큰이 주어진다. 

특이한 점은 SEP 토큰이 주어지지 않아서 답/질문을 가르는 delimeter가 없음. 이것을 맞추도록 학습시키는 것.

![Untitled](source/Untitled%202.png)

The top results from the SQuAD leaderboard do not have up-to-date public system descriptions available,11 and are allowed to use any public data when training their systems. We therefore use modest data augmentation in our system by first fine-tuning on TriviaQA (Joshi et al., 2017) befor fine-tuning on SQuAD.

In fact, our single BERT model outperforms the top ensemble sys- tem in terms of F1 score.

Without TriviaQA fine-uning data, we only lose 0.1-0.4 F1, still outper- forming all existing systems by a wide margin.

v2.0 은 지문에서 답이 주어지지 않는 경우가 있다. 해당 경우, CLS에 표시하도록

![Untitled](source/Untitled%203.png)

## useful links

[https://velog.io/@xuio/NLP-논문리뷰-BERT-Pre-training-of-Deep-Bidirectional-Transformers-forLanguage-Understanding-상편](https://velog.io/@xuio/NLP-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-BERT-Pre-training-of-Deep-Bidirectional-Transformers-forLanguage-Understanding-%EC%83%81%ED%8E%B8)