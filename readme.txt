hexo登录流程

进入git bash

cd _workspace/ianZzzzzz.github.io/  # 进入开发环境

hexo new "Post Test" # 创建新博文 生成PostTest.md文件

git status

git add -A         # add

git commit -m ":art:create a new post"   # commit

hexo d -g   #  生成项目文件

hexo g -d   # 部署PostTest.md文件

git branch # 分支
	master # 分支名

git checkout master # 转到分支上操作

git merge develop # 和develop分支合并

git push origin master --force

hexo g -d 


---------------
hexo s # 启动本地服务
