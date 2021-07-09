# Python代码仓库模板

![GitHub forks](https://img.shields.io/github/forks/GT-ZhangAcer/PythonRepository-Template?style=for-the-badge) ![GitHub Repo stars](https://img.shields.io/github/stars/GT-ZhangAcer/PythonRepository-Template?style=for-the-badge) 

这是一个简单的迁移模板，使用者只需在[模板仓库](https://github.com/GT-ZhangAcer/PythonRepository-Template)中点击[use this template](https://github.com/GT-ZhangAcer/PythonRepository-Template/generate)即可创建属于自己的具备前端页面空白Paddle项目。


## 项目结构

### Master分支（Default）
该分支为主要的开发分支，与项目有关的说明和代码文件可放置于此，在仓库被访问时默认展示该分支。
```
-|
--PaddleOCR  项目运行代码
--data_process  训练数据离线预处理代码
--inference  推理模型
--checkpoint 推理模型对应的checkpoint文件
--Dockerfile  docker文件
--charset.txt  模型训练和测试使用的字典表
--LICENSE   开源协议文件，默认为MIT开源协议。
--README.md 项目说明文件，可使用Markdowm编辑器进行编辑。
--requirements.txt Python项目依赖列表
```  
