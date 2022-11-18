# 포토샵은 이제 그만, 대세는 MaskGAN이다
### [2022 경희대학교 소프트웨어페스티벌](https://swf.khu.ac.kr/)

## 개요

현대 사회에 보정 어플은 상당히 발전되어있다. 하지만 기능이 뷰티에 초점이 맞춰져 있다는 한계점이 있다. 따라서 본 프로젝트에서는 얼굴 커스터마이징뿐만 아니라 헤어스타일, 귀걸이, 옷, 모자, 안경 등 다양하게 적용할 수 있는 프로젝트를 구현하고자 하였다. MaskGAN을 이용하여 사용자 친화적인 GUI 형태의 프로그램을 오픈된 코드와 논문을 바탕으로 학습 코드와 최종 Demo를 구현하였다. 

## Dataset
* CelebA 데이터셋 30000장
* 팀원별 사진 4장 추가 학습

<br>

![CelebA Data](https://ifh.cc/g/kBXLRc.jpg)

## Model & Tools
* Pix2PixHD 
* PyQt5
* Pytorch 
* numpy
* opencv-python

## 실행 사진

![ezgif com-gif-maker (7)](https://user-images.githubusercontent.com/83739271/201490887-96b5ca70-53be-40af-b803-2a7f4d157a46.gif)
</br>

![이원준](https://ifh.cc/g/qXQFdz.jpg)
<div align="center">
* 귀 모양 변경, 귀걸이 생성 결과
</div>

</br>

            
![이승현](https://ifh.cc/g/jj1PMM.jpg)

<div align="center">
* 헤어스타일 변경, 귀 모양 변경, 코 모양 변경, 입 모양 변경
</div>

</br>

![명승준](https://ifh.cc/g/FbNsWs.jpg)

<div align="center">
* 목 모양 변경, 입술 모양 변경, 코 모양 변경 
</div>

</br>

![임달홍](https://ifh.cc/g/x8gv4w.jpg)

<div align="center">
* 헤어스타일 변경, 옷 생성, 귀걸이 생성
</div>

</br>

* 상세한 실행 과정은 아래 영상을 참고
   * [Demo Video](https://youtu.be/W-l0jjOcYGE)

* 기능은 총 18가지가 있다.
*  각 기능에 맞춰 얼굴 커스터마이징이 가능하다.
    * Skin 
    * nose 
    * Eyeglass
    * Left Eye, Right Eye
    * Left Eyebrow, Right Eyebrow
    * Left ear, Right ear
    * Mouth, Upper Lib, Lower Lib
    * Hair
    * Hat
    * Earring
    * Necklace
    * Neck
    * Cloth

## 실행 방법

* Pytorch 1.13.0+cu116
* torchvision 0.14.0+cu116
```python 
!python demo.py #MaskGAN_demo
```
* MaskGAN을 실행하기 위해서 face_mask 이미지와 원본 이미지를 필요로 한다.

* fase_parsing을 통해 mask된 이미지 도출
```python
!bash run_test.sh #gpu_num #face_parsing
```
* fase_parsing 과정
    * face_parsing/Data_preprocessing/g_partition.py 실행 -> train/test/val 폴더 생성

    * test_img 폴더에 fase_parsing 수행할 데이터셋 첨부

    * 데이터셋의 수에 따라 parameter.py를 test_size를 수정해야하고 batch_size를 조절하거나 아래 코드를 수정해야 할 필요가 있다.
    ```python
    #face_parsing/tester.py
    path = test_paths[i * self.batch_size + j]
    ```


## 수정 사항

* CelebA 데이터로 사전 학습된 MaskGAN 모델에 각 팀원들 사진과 mask 이미지를 추가하여 continue_train을 수행하였다.

* [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD) train 코드를 참고하여 MaskGAN에 적합하게 train 코드를 구현하여 진행하였다.

- - - - -

* Data Dictionary를 불러오는데 이미지 input값에 라벨 값들을 수정하여 추가해주었다.
```python
#train.py
#line 88
inter_label_1, label = data['inter_label_1'], data['label']
inter_label_2, label_ref = data['inter_label_2'], data['label_ref']
image, image_ref, path, path_ref = data['image'], data['image_ref'], data['path'], data['path_ref'] 
```

<br>

* 이미지 visualizer에 대한 코드는 불필요하여 제거하였다.
* model/network.py가 CPU연산으로 돌아가서 GPU로 돌아가게 수정하는 코드를 추가하였다.
```python
#train.py 
#line 52
model = create_model(opt)
device = 'cuda'
model = model.to(device)
```

* 그 외 세부적인 수정사항들은 train.py를 참고

- - - - -

* 참고한 코드의 라이브러리 버전이 구 버전이여서 현재 최신 버전의 라이브러리 버전에 맞게 코드를 수정하여 진행하였다. 

* 실행시키면서 발생한 문제점들과 구 버전 라이브러리들이나 코드 속 에러들을 수정한 사항들은 다음과 같다.
```python
#이미지를 불러오는데 아래 코드에서 오류가 발생하는 경우가 있다.
self.img = mat_img.copy()
```
* cuda 미설치, 이미지 인풋 사이즈 확인
* 이미지 경로에 한글이 포함되어 있으면 호출 안됨-> MaskGAN_demo/samples에 이미지를 추가하여 다시 이미지 로드

<br>

```python
#demo.py 
#line - 206
result = result.cpu().numpy() 
-> result = result.detach().cpu().numpy()
```
* 연산 기록으로 부터 분리한 tensor을 반환하는 메소드인 detach()를 추가하라는 오류 메세지가 출력

<br>

```python
#line - 209
qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)
-> qim = QImage(result.data.tobytes(), result.shape[1], result.shape[0], 3*result.strides[0], QImage.Format_RGB888)
```
* argument 1 has unexpected type 'memoryview' 오류가 발생.
* Qimage 인풋 값에 맞게 result.data에 .tobytes() 추가. 컬러 이미지를 결과로 도출하기 위해 8바이트가 3채널 이므로 이미지의 폭에 3을 곱해서 표시 추가

<br>

```python
#face_parsing/tester.py
#generate_label_plain input값에 맞게 파라미터 추가
#def generate_label(inputs, imsize)
labels_predict_plain = generate_label_plain(labels_predict)
labels_predict_color = generate_label(labels_predict)
-> 
labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
labels_predict_color = generate_label(labels_predict, self.imsize)
```
* generate_label_plain input값에 맞게 파라미터 추가 [def generate_label(inputs, imsize)]

<br>

```python
#face_parsing/tester.py
path = test_paths[i * self.batch_size + j]
```
* batch_size를 학습 데이터의 수에 맞게 parameter.py에서 수정해야 한다. 그렇지 않으면 위 코드에 의해 out of list 오류가 발생한다. 

<br>

## 결론 및 문제점

* fase_parsing을 수행하기 위해 CelebA Dataset보다 적합하고 많은 데이터셋 확보에 어려움과 학습 환경의 제한의 어려움 있어서 pretrained_model로 수행해야 했다. 

* 참고 깃허브와 논문에서 train에 대한 코드를 제공하지 않을 뿐만 아니라 참고할 만한 코드가 한정적이라 많은 시간이 소요되었다.

* 사전 학습 모델을 이용하여 image to image translation을 수행하기 때문에 학습되지 않은 데이터를 넣고 커스터마이징을 할 시 잘못 학습된 결과물이 나오는 경우가 종종 있다. 따라서 위 프로그램을 통해 올바른 아웃풋을 얻기 위해서는 원하는 이미지의 모델 학습이 필수적이다.

* 모델 학습을 진행하고 참고 깃허브와 논문에서 제공한 pretrained_model만 사용할 시 CelebA 데이터셋을 제외한 이미지에 대해서는 정확도가 떨어지는 결과값이 출력된다. 따라서 train 코드를 추가하여 continue_train을 수행하여 참가 팀원의 사진과 mask를 학습하였다.

## Collaborator
* [Seung Joon Myeong](http://github.com/MSJ99)
* [Seung Hyun Lee](https://github.com/dltmdgus0220)
* [Dal Hong Lim](https://github.com/limdalhong)

## Reference
* [MaskGAN : Towards Diverse and Interactive Facial Image Manipulation](https://arxiv.org/abs/1907.11922)
* [MaskGAN github](https://github.com/switchablenorms/CelebAMask-HQ)
* [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
```
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
