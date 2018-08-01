## Amazon 클라우드 (AWS) 를 활용한 GPU 실습

윈도우즈의 경우 OpenSSH 사용: [LINK](https://www.admfactory.com/how-to-install-openssh-on-windows/)


For ubuntu **login** command is
> ssh -i my-pem-file.pem ubuntu@my-ec2-instance-address

For RHEL it is
> ssh -i my-pem-file.pem root@my-ec2-instance-address

Connecting to an ec2 instance does not require a password, it would require only a pem file and this is how you connect to it
> ssh -i my-pem-file.pem ec2-user@my-instance-address
