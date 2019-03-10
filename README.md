## 0. 스터디 소개

![GPP](C:\Users\lalat\Pictures\GPP.jpg)

- 팀명 : GP (GoldenPass: 골든패스)
- 팀원 : 김대영, 김태우, 서창원, 최종현, 한성일
- History
  - 2018.07.04 구글 머신러닝 스터디 잼 입문반 스터디 결성
  - 2018.07~08. 구글 머신러닝 스터디 잼 입문반 수료
  - 2018.09. Coursera course로 DL 스터디
  - 2018.10. 구글 머신러닝 스터디 잼 심화반 수료
  - 2018.11~12. Kaggle, DL 관련 책으로 스터디
  - 2018.01~02. HeLP-Challenge 참가 (2-2. 주제부분 3위/10팀)
  - 2018.02~  [Facebook Developer Circle: Seoul](https://www.facebook.com/groups/DevCSeoul/)에서 지원 받아 **Spark Plus**에서 스터디 중!! 



## 1. 챌린지 소개



![1552216758231](C:\Users\lalat\AppData\Roaming\Typora\typora-user-images\1552216758231.png)

본 `HeLP Challenge`는 서울아산병원이 주최하는 의료인공지능 개발 콘테스트다. [관련 링크](http://bigdata.amc.seoul.kr/asan/depts/bigdata/K/bbsDetail.do?menuId=4319&contentId=264622&versionNo=2)

- 대회 목적 : 인공지능 개발자들이 서울 아산병원의 의료데이터에 접근하여 연구할 수 있는 플랫폼을 제공함으로써 세계적인 의료인공지능기술 개발에 기여하고자 함.

- 참가자격 : 의료인공지능기술개발에 관심이 있는 개인, 대학, 스타트업, 연구기관, 기업
- 콘테스트 내용 
  - 1-1 뇌종양 MRI에서 뇌종양 영역 분할
  - 1-2 심장 CT에서 right&left ventricle chambers, left ventricle myocardium, and papillary muscle 분할
  - 2-1 뇌경색발생과 MRI 촬영 사이의 시간 추정
  - **2-2 병리조직 슬라이드에서 breast cancer metastasis 여부 판정**
- 우리 팀이 참가한 주제는 `2-2 병리조직 슬라이드에서 breast cancer metastasis 여부 판정`이다.
- `Contest 2-2. Breast cancer classification on frozen pathology` 주제와 관련된 [description wiki](Contest 2-2. Breast cancer classification on frozen pathology) 
- 기간 : 2019.01.21.월 ~ 2019.03.03.월  6주간 진행



## 2. 진행 과정

- 사전조사 : [breast_cancer_challenges_tu.xlsx](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/breast_cancer_challenges_tu.xlsx)의 `S` sheet 참고

- 진행과정 : [breast_cancer_challenges_tu.xlsx](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/breast_cancer_challenges_tu.xlsx)의 `Sc` sheet 참고

  - 2018.12.31 ~ 2019.01.06 (1주차) : 사전조사 완료

  - 2019.01.07 ~ 2019.01.13 (2주차) : Windows 환경에 Openslide package 설치, 관련 git 조사

  - 2019.01.14 ~ 2019.01.20 (3주차) : 설명회 참가. (거의 아무것도 안함)

  - 2019.01.21 ~ 2019.01.27 (4주차) : **대회 시작.**

    - Openslide Package 뜯어보기
    - Data preprocessing
    - Whole process coding
    - Data analysis 
    - Docker 공부, Finished Image 생성

  - 2019.01.28 ~ 2019.02.03 (5주차) : 대충 모델이 완성되어 거의 아무것도 안함..

  - 2019.02.03 ~ 2019.02.10 (6주차) : **설날.**

    - 도커로 학습되는지 확인
    - 도커환경에서 돌릴 때 발생한 문제점들 하나씩 해결 (image build, docker file structure, package download, data preprocessing, memory, file handle etc..)

  - 2019.02.11 ~ 2019.02.17 (7주차) : 도커 학습 성공

    - 02.10. 첫 도커 학습 성공, 4시간 학습, Phase 1 score : 0.86
    - training set sample size, model (simple, unet, inception) 변경해가며 적용
    - data aug 빼면서 30시간 학습 (성능저하)
    - data aug 추가해보면서 학습시킴, 다양한 data aug 실험
    - 30 시간 학습 --> score 0.93으로 증가. epoch, 

  - 2019.02.18 ~ 2019.02.24 (8주차) : **Phase2 Open**: 새로운 data set

    - Phase 2 submit에 맞게 inference 수정(그 전까지 학습한 model 활용 불가하게 됨..)
    - 02.19. Phase 2 첫 score 0.66
    - Phase1 score와 Phase2 score의 차이가 큼. sampling , 학습방법 잘못됐다는거 인지
    - 02.21. sampling method 수정

  - 2019.02.25 ~ 2019.03.03 

    - 02.25~27. 학습속도가 너무 느려 image 저장하고 data 저장하는 방식으로 변경, 30시간 학습, 성능 저하..

    - learning decay (keras call back 활용) 

    - 02.21에 변경한 모델로 60시간 정도 학습하는 걸로 마무리 (20000 samples 20 epochs) ( Phase 1 score 0.86 -> Phase 2 score 0.75)

    - 03.03. ensemble 노가다. Phase2 `0.76` 마무리 (3위 / 10 팀)

      

## 3. 관련 기술 소개

- 도커 
  - [관련 설명](https://taeu.github.io/tech/%EB%8F%84%EC%BB%A4-Windows-%ED%99%98%EA%B2%BD%EC%97%90%EC%84%9C-Docker-%ED%99%9C%EC%9A%A9/) 
- 전처리
  - openslide : [openslideTest.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/openslideTest.ipynb), [openslideTest_2.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/openslideTest_2.ipynb)
  - 전처리 :  

- 학습

  - 관련 설명 : 
    - 전반적인 설명 : [description_whole_pr_0121.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/description_whole_pr_0121.ipynb)

  ![1552216167569](C:\Users\lalat\AppData\Roaming\Typora\typora-user-images\1552216167569.png)

  - 네트워크
    - simple : [model_simple_1.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/model_simple_1.ipynb)
    - unet : [model_unet_1.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/model_unet_1.ipynb)
    - inception : [model_inception_1.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/model_inception_1.ipynb)
  - data aug 확인사항 : [data_augmentation_keras_test.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/data_augmentation_keras_test.ipynb)
  - 샘플링까지 제대로
    -  [docker-simple-final](https://github.com/Taeu/HeLP-Challenge-Goldenpass/tree/master/docker-simple-final)
    - [docker-inception-3-4](https://github.com/Taeu/HeLP-Challenge-Goldenpass/tree/master/docker-inception-3-4)
    - (속도 향상한다고 잘못 sampling 했었던 모델) [docker-simple-3](https://github.com/Taeu/HeLP-Challenge-Goldenpass/tree/master/docker-simple-3)



## 3. 결과

```Contest 2-2. Breast cancer classification on frozen pathology```  3위 / 10 팀

![1552218804890](C:\Users\lalat\AppData\Roaming\Typora\typora-user-images\1552218804890.png)



![1552218828793](C:\Users\lalat\AppData\Roaming\Typora\typora-user-images\1552218828793.png)



 **(Phase 2로 시상)**



## 4. 후기









