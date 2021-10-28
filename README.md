
# GPU를 이용한 병렬 프로그래밍 (CUDA 병렬 프로그래밍)
## High Performance Parallel Programming with Nvidia GPU

### 이정근, 한림대학교 소프트웨어융합대학 교수 (Lecturer: Jeong-Gun Lee, Hallym University)
### Web: www.onchip.net

### 주요 자료 (Main Teaching Materials)
- 강의 자료는 [PPT 디렉토리](https://github.com/jeonggunlee/CUDATeaching/tree/master/PPTs)볼수 있습니다. (All the lecture materials can be found in [PPT Directory](https://github.com/jeonggunlee/CUDATeaching/tree/master/PPTs))
- 실습 자료는 [01_a_lab](https://github.com/jeonggunlee/CUDATeaching/tree/master/01_cuda_lab), [02_a_lab](https://github.com/jeonggunlee/CUDATeaching/tree/master/02_cuda_lab), 및 [03_a_lab](https://github.com/jeonggunlee/CUDATeaching/tree/master/03_cuda_lab)을 참조하시기 바랍니다. (Lab materials can be found in [01_a_lab](https://github.com/jeonggunlee/CUDATeaching/tree/master/01_cuda_lab), [02_a_lab](https://github.com/jeonggunlee/CUDATeaching/tree/master/02_cuda_lab), and [03_a_lab](https://github.com/jeonggunlee/CUDATeaching/tree/master/03_cuda_lab))
- Most of lab materials are developed with Google Colab environment. So you can do the lab practices with web browser in your computer without any GPU.
- [짧은 Youtube 동영상 강의](https://www.youtube.com/playlist?list=PLKZ28p5qq0DGLcO6QZdMSG_jsprRtG15C) (Short lecture movie clips can be found in [Short Introduction to CUDA in Youtube](https://www.youtube.com/playlist?list=PLKZ28p5qq0DGLcO6QZdMSG_jsprRtG15C))


## Schedule (스케쥴)
  - **Day 1**
    -	병렬프로그래밍소개 ([PPT](./PPTs/Robot_01_Intro. Parallel Computing.pdf), [pthread / openmp / avx](./01_cuda_lab))
    -	GPU 병렬프로그래밍 기초
    -	GPU 아키텍쳐
    - GPU 병렬프로그래밍: 실습
      - colab.research.goole.com에서 GPU 활용하기 ([hello_CUDA.ipynb](./hello_CUDA.ipynb))
      - [Google Drive와 연동하기](https://github.com/jeonggunlee/CUDATeaching/blob/master/colab_gdrive.ipynb)
      - [Udacity GPU Programming 강좌 예제 파일](https://github.com/jeonggunlee/cs344/)
    
  - **Day 2**
    -	GPU 프로그래밍: 최적화
    -	GPU 프로그래밍: 최적화 실습
    -	Parallel Transpose 최적화
    -	Parallel Reduction 최적화
    -	CUDA 최적화 실습
    
  - **Day 3**
    -	병렬 CUDA 벡터곱 최적화 (2시간)
    -	Host-GPU 인터페이싱: Streams
    -	Host-GPU 인터페이싱: Streams 실습
    -	개발사례: GPU 기반 초음파영상처리

*  *  *
![Schedule](./images/schedule.JPG)
*  *  *

## REFERENCES (참조Sites)
  -Good-to-See Source Example: https://github.com/jeonggunlee/cs344
  - CUDA Sample Directory: C:\ProgramData\NVIDIA Corporation\CUDA Samples
  - CUDA 최고의강좌! 강추! Udacity [Intro to Parallel Programming](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2)
  - Udacity [High Performance Computer Architecture](https://www.youtube.com/watch?v=tawb_aeYQ2g&list=PLAwxTw4SYaPmqpjgrmf4-DGlaeV0om4iP&index=1)
  - Udacity [High Performance Computing](https://www.youtube.com/watch?v=grD5en6_IiQ&list=PLAwxTw4SYaPk8NaXIiFQXWK6VPnrtMRXC)
  - [CUDA LECTURE](https://www.youtube.com/watch?v=sxhvmTveO2A) - Oklahoma State University
  - 코딩 실습을 위한 [클라우드 설정(AWS)](https://github.com/jeonggunlee/CUDATeaching/blob/master/gpu4cloud.md) 
