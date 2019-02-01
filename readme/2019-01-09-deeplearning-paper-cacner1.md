---
layout: post
title: "[논문] Detecting Cancer Metastases on Gigapixel Pathology Images 분석"
category: paper
tags: dl paper breastCancer healthcare medical
comments: true
img: medical4.jpg 
---



# Detecting Cancer Metastases (암 전이 발견 )



 이번 1월 20일부터 서울 아산병원에서 주최하는 의료인공지능 개발 콘테스트에 참가하게 되었다. 총 4개의 주제 중 콘테스트2-2, **"병리조직 슬라이드에서 유방암 전이의 여부를 판정"** 하는 task를 밭았다. 이와 관련해서 관련 자료를 찾아보던 중 Camelyon16에서 "detection of cancer metastasis"의 같은 주제로 대회를 열었었고, 그 중 우수작 및 관련 논문을 더 탐색하다 오늘 글의 논문인 **"Detecting Cancer Metastases on Gigapixel Pathology Images"**를 발견했다. 일단 콘테스트 전 간단히 모델을 만들어볼 필요를 느껴 이 논문을 읽게 되었다.



### 참고자료

- [논문](https://arxiv.org/pdf/1703.02442.pdf)




# 0. Abstract

매년, 미국에서는 230,000 명의 유방암환자에 대한 치료의 결정은 유방으로부터 다른 조직으로 전이가 되었는지의 여부에 전적으로 달려있다. 병리학자들은 전이의 발견을 위해 많은 시간과 노력을 쏟지만 이 과정은 여전히 강도높은 노동이며 error, 오진이 나오기도 한다. 

우리는 이런 task를 CNN의 구조와 Camelyon16에서 sota의 결과들을  가져와 97% 이상의 AUC 점수와 카Camelyon16의 training set에서 2개의 잘못 라벨링된 데이터(086,144)를 발견했다. 또 우리의 접근은 FN(false negative)를 줄였다.



# 1. Intruduction



이 논문의 핵심 요약은 다음과 같다.

- CNN , Inception V3
- careful image patch sampling
- careful data augmentation



> We also found that several approaches yielded "NO" benefits

- (1) multi-scale apporach
- (2) pre-training model the model on ImageNet image Recognition
- (3) color normalization



# 2. Methods

### 2-1. Inception V3

인셉션 V3 구조를 사용, input size는 299x299. 각 인풋 패치에서 중심 128 x 128 영역의 라벨을 예측한다. 각 패치의 라벨들은 중심 영역에서 적어도 한 픽셀이 종양이라고 라벨링(annotated)되어 있다면 종양이라고 라벨링한다.

또, 파라미터 수(filter 수)를 줄여 가며 # of parameter가 주는 영향을 실험했다(depth_multiplier=0.1 In TF). 멀티 스케일은 별로 효과가 없었으므로 우리는 2개의 크기(멀티스케일된 크기)만 사용.



### 2-2. Sampling

슬라이드당 90,000개(median)의 'normal' 패치와 2,000(median)개의 'tumor'종양 패치가 나왔는데 대략 이 둘의 비율은 2%밖에 되지 않는다. 따라서 이런 언밸런스를 해결하기 위해 sampling을 필수적으로 거치는데, 

- 먼저 'normal'과 'tumor'를 같은 확률로 뽑고
- 그 라벨의 패치들을 가지고 있는 슬라이드를 랜덤하게 뽑고
- 그 슬라이드에서 패치들을 샘플링한다.

(훈련중에 나타나는 패치수들을 한정적으로 만들지 않기 위해) 



### 2-3. Data Augmentations

- 4 multiples of 90˚ rotations. + left-right flip (8 orientations)
- perturb color : (maximum delta)
  - brighteness 64/255,
  - saturation 0.25,
  - hue 0.04
  - contrast 0.75 

- Jitter : up to 8 pixels.
- pixel values clipped [0,1], and scaled [-1,1]



### 2-4. Prediction

>We run inference across the slide in a sliding window with a stride of 128 to match the center region’s size.



- obtain predictions for each of the 8 orientations (rotate, left-right flip), and avg.



### 2-5. Implementations Details

- batch size = 32
- RMSProp , momentum 0.9, decay 0.9,  $\epsilon$ = 1.0
- initial lr = 0.05, 0.5 decay every 2 M ex.
- for refining a model pretrained on ImageNet, initial lr = 0.002



( FROC Metric check)