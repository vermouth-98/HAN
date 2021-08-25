# HAN
## cấu trúc

![cautruc](https://user-images.githubusercontent.com/76995105/130818355-fbe8c8bd-eab7-4d41-9b84-3c01e2efa255.png)

Trong đó residual group theo paper gồm 20 RCAB và gồm có 10 khổi residual block
tiếp theo là cấu trúc của layer attention module

![LAM_module](https://user-images.githubusercontent.com/76995105/130819032-631bd051-873c-40ed-adb3-21474a5d6773.png)

Cấu trúc của channel-spatial attention

![CSA_module](https://user-images.githubusercontent.com/76995105/130819716-5925bf5f-a1c4-4a9a-8b66-0712fc903de3.png)

## Dữ liệu
dữ liệu đầu vào là các ảnh LR( low quaility) and HR (high) từ bộ dữ liệu DIV2K 
Ảnh được crop đưa về kích thước 64x64
augmentation bao gồm horizontall fill, random rotation 90,180 và 270
và ảnh được chuyển về dạng YCbCr space

## hàm loss function
trong model hàm loss được sử dụng là MSEloss hoặc L1Loss

## load model
download pretrain model in here:
https://drive.google.com/drive/folders/17cLcPCDLuBV5_5-ngd0vXIDp6rebIMG1
## vidu train
ảnh gốc kích thước 48x48

![Untitled](https://user-images.githubusercontent.com/76995105/130820888-f82ad4aa-9071-4441-85fd-19deb0a9b06d.png)

ảnh sau khi qua model 384x384

![img (1)](https://user-images.githubusercontent.com/76995105/130821018-af79c8d6-8cd1-4852-b99d-50d6ea1cb96c.png)
