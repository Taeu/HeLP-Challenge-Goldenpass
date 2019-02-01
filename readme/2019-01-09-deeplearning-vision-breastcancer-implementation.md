---
layout: post
title: "[DL] 유방암 진단 네트워크 구현"
category: vision
tags: dl vision breastCancer healthcare medical
comments: true
img: medical4.jpg 
---



# 유방암 진단 딥러닝 모델 구현

본 글은 서울 아산병원과 카카오브레인에서 주최하는 HeLP Challenge 의 2-2. Breast cancer Metastasis Detection 의 대회 log를 기록한 것이다. 

## 참고자료

- [kaggle 데이터셋](https://www.kaggle.com/questions-and-answers/50144)
- [Camelyon16](https://camelyon16.grand-challenge.org/)

---



# 0. 사전 조사 과정

### 1. Data

___



- breast cancer data(Camelyon16) - ok
- visualization tool : ASAP ([Camelyon16](https://camelyon16.grand-challenge.org/) recommend to use) - ok
- OpenSlide : https://pypi.org/project/openslide-python/ - ok
- OpenSlide_installation : https://github.com/openslide/openslide-winbuild (Execute: 부분에서 막힘, 안해도됨) -ok
- labeling the data (using xml file which has annotated inform..,) - try
- convert tif img to npy - try (WSI)
- (later-inference) npy to tf img - try
- (if have time) localization, -(비교적 easy,)
- h5 file : https://github.com/basveeling/pcam (나중에)
- 일단 data 잘 aug + sampling 했다고 가정하고 다음 [git - HMS-MIT method1](https://github.com/zhudaoruyi/camelyon/blob/master/main.py) 참고 - ok
- data augmentation method search (jitter part + ) - try
- ostu method search - ... 
- consider **sampling part** more + **extract patches** - try
  - https://github.com/arjunvekariyagithub/camelyon16-grand-challenge
  - https://github.com/baidu-research/NCRF





### 2. Model

- Inception V3 keras implementation
  - input size
  - center predict ( 128 x 128 )
  - predict size
- sampling ++



### 3. Insight materials



- [kaggle - 한국분 우승방법 소개]( https://www.kaggle.com/pudae81 )

- [카카오브레인 글 관련](http://www.kakaobrain.com/blog/48?fbclid=IwAR2dd8v0zWiYwuEtnG2GXI15OvB4cLOgA5BV4uL-LJErqfUklRAYVgKu1cA)

- 논문도 좋고





### 4. Challenge Preview



- discussion : https://www.synapse.org/#!Synapse:syn15569329/discussion/default
- description : https://www.synapse.org/#!Synapse:syn15569329/wiki/582435c
- Docker 이해 필수
- Data -> 이미 조직이 있는 위치 나옴. 배경 분리하는 작업은 필요 없고, 적당한 patch로 자르는 sliding window 방법으로 구현하면 될 것으로 보임. 이와 관련된 거는, fast-R CNN 계열 보면 될 듯.

  - Patch 자르고 augmentation 까지 저장된 파일이 메모리 상에 위치해야하는 것 같음. 그 GPU는 T4니까 6-9GB 정도일 듯. 배치사이즈 좀 줄이고 셔플 필수
- model 앙상블 하면 좋을 듯.
- external dataset 확보 필요, 아까 kaggle 에 언급한 2군데 들어가보기
- openSlide 이용해서 데이터 처리하는 거 필수로 해보기 : https://github.com/deroneriksson/python-wsi-preprocessing
- 모델 weight 저장법
- 질문할 것 : 

  - 1. data annotation 이 어떻게 되어 있는지. (가장 중요, 픽셀단위인지 아닌지부터) -ok
    2. data shape 관련해서도 질문, 아마 슬라이드로 나눠서 저장된거겠지?  아 그건또 아니네 생각해보면( 왜 240 x 240 x 170인지랑 2개 사이즈가 있다고 했는데 그것도) -ok open slide
    3. 서버 상에서 data view가 가능한지 -ok
    4. 샘플 하나만 보여줄 수는 없는지 -ok
    5. **시도횟수는 무한인지.** (아마)
    6. output과 관련해서 추가로 알아야할 사항은 무엇인지도





### 5. challenge introduction



- 녹화자료는 공유됨.
- 수술시 림프절 슬라이드 진단을 통해 겨드랑이쪽 림프절을 수술해야할지의 여부 결정
- 열악한 환경에서 AI가 진단해준다면 좋음 (false negative : 19~42%)
- 진단 정확성, 5분 안의 limit

- Dataset
  - .mrxs 
  - mask issue : 흰색이 그려진 마스크, tumor,
  - 원본과 매칭 필요
  - level  patch 크기 , x, y
  - prob
  - pixel by pixel 
  - 448,448 (patch size)
  - 
- library
- Measure
- Ex (유의해야할 예제들) 



- Docker
- model 에 data  출력 후 재 inference
- ./train.sh 
- 처음 확인하는 작업 실행
- ./inference.sh 파일 있어야함
- csv header 출력 x
- inference 까지 완료되어야 함.
- 시간싸움인듯.
- https://help-khidi.kakaobrain.com/
- docker img 중에 gpu 활용 가능한 img 들이있다.(hub)
- 중간에 gpu 썼는지,
- Q&A



# 1. Whole Process Sketch

Breast Cancer Metastasis Detection Model의 전반적인 스케치는 다음과 같다.

1.  All tissue patches sampling
2.  Train Data Generate
3.  Network Define
4.  Train
5.  Predict



다음은 각 과정에서 고려했던 혹은 고려해야할 세부사항들이다.



## [1] All tissue patches sampling

1-1. Sampling 전 슬라이드 데이터 분석

- data analysis python 을 작성해 docker 이미지 올린 결과 아래와 같음

 - num of slide = 157
 - total num of patches(patch size = 256, all tumor 기준) = 4,000,000
 - total num of tumor patches =  600,000 
 - non tumor : tumor = 5.8 : 1 

추가적으로 고려(참고)해야할 사항

- negative mining
  -  sampling, 
  - ob det method ( refinet or yolo or other ex)) 
  - google 17 paper
  - 5.8 : 1 도 나쁘지 않은 것 같음
  - patches 기준 descending order -> negative 1 + positive 2 로 구성 (초반 sample들은 3:1정도 구성하게 됨)
- memory size 
- time 
  - 위 두 문제는 negative 1 : positive 2 로 구성하면 됨. train data 생성 전 # of samples(train data)를 len(all_tissue_sample)로 잡으면 될 것 같음



1-2. 구현 :```find_patches_from_slide()``` 

- tissue check 
  - grey 변환 검정색 걸러내고
  - ostu method 통해 남은 흰색 배경 걸러냄
- tumor check
  - positive 경우에만 고려
  - mask가 이미 1/16 이 되었기 때문에 mask는 patch_size/16 만큼만, 원 slide는 lv 0에서 patch_size 만큼 level down. (256 기준 slide는 level 8, truth_mask 는 level 4)
  - all tumor 만 고려. mask된 비율이 차이가나므로 원본 데이터와 비교했을 때 정확하게 masking 되지 않는 문제 발생. 따라서 all tumor(모든 영역이 tumor 인 patch만 고려해줘야함)
  - 256 기준으로 약 약 25개의 slide의 tumor patch 개수가 100개 미만, 그 중 10개정도는 0개. 이 부분에 대해서는 128 size로 patch를 조정해 tumor patch를 더 확보하는 방안도 있음 

- 각 기록된 것의 loc 가져 오기
- return all_tissue_samples



## [2] Train Data Generate

구현 : `img_gen()`

- X: (batch_size, patch_size, patch_size, 3)
- Y: (batch_size, patch_size, patch_size, 2)
- Data augmentation 구현 해야함
- yield 로 메모리 overhead 방지



## [3] Network



- simple U-Net (+ skip connection 구현)
- Inception-v3 (Google paper 17)
- DenseNet 
- (like 2018.12 Protein detection comp's 1st solution)
- parameter 고려
- hy.par. 고려



## [4] Train

- num_samples = len(all_tissue_samples)
- batch_size = 50~100 (OOM check from 1080 ti)
- train : validation = 8:2 or 9:1
- model save path : /data/model
- hyper parameter tuning
- parameter size reduction





## [5] Predict

- center of img (google 17 paper)
- ovelap 고려











 















​    	
