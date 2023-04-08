# Git学习（23/4/8学习）

## Git是什么

是目前世界上最先进的分布式版本控制系统，可以记录每次对文件的改动，实现多人协作工作

## Git的优势

不用联网，假设正在一个没有网络连接的地方，但是你需要修改你的代码并且保留修改历史记录。如果你使用其他的版本控制系统，比如 SVN 或者 Perforce，你可能需要等到有网络连接之后再提交你的修改。而如果你使用 Git，你可以在本地进行修改并且提交到本地的代码仓库中，不需要联网，这使得 Git 在离线环境下非常方便。

## git的安装

安装教程：[(40条消息) Git 详细安装教程（详解 Git 安装过程的每一个步骤）_git安装_mukes的博客-CSDN博客](https://blog.csdn.net/mukes/article/details/115693833?ops_request_misc=%7B%22request%5Fid%22%3A%22168091658016800227422842%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=168091658016800227422842&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-115693833-null-null.142^v82^insert_down38,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=安装git&spm=1018.2226.3001.4187)

## 本地仓库

#### 初始化一个本地仓库

在想要存储的文件夹中启动bash，输入`$ git init`即可初始化一个空的本地仓库

#### 将文件添加至本地仓库

1. 首先使用vscode在本地仓库创建一个文件

2. 再在bash中使用命令`$ git add "fileName"`将其添加至仓库

3. 最后使用命令`$ git commit -m "Write a readme file"` 提交给仓库并添加对该文件的说明，说明必须有实际意义

4. 两个命令

   `git status`告诉你有文件被修改过，用`git diff`可以查看修改内容。

[^Tips：add和commit命令分开使用，是因为可以添加多个文件，一次提交]: 

##### 版本之间切换

- 使用`$ git reset --hard` 【版本号/HEAD^】移动或者回退到相应的版本
- 可以使用`$ git log` 查看历史提交记录
- 使用`$ git reflog` 查看命令历史

##### 工作区和暂存区

工作区就是自己创建的一个存放相关文件的文件夹，例如我的本地创建的WorkSpaceGit文件夹

暂存区

工作区有一个隐藏目录`.git`，这个不算工作区，而是Git的版本库

Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的暂存区，还有Git为我们自动创建的第一个分支`master`，以及指向`master`的一个指针叫`HEAD`

![image-20230408110051000](C:\Users\Kivi\AppData\Roaming\Typora\typora-user-images\image-20230408110051000.png)

简单理解为需要提交的文件修改通通放到暂存区，然后，一次性提交暂存区的所有修改

`git add`命令实际上就是把要提交的所有修改放到暂存区（Stage），然后，执行`git commit`就可以一次性把暂存区的所有修改提交到分支。

##### 管理修改

只有add过的内容提交过后才能被commit提交

##### 撤销修改

- 还没有add的使用`$ git checkout -- file`可以丢弃工作区的修改
- 使用了add的使用`$ git reset HEAD <file>`可以把暂存区的修改撤销掉（unstage），重新放回工作区
- 如果commit了，使用`$ git reset --hard <版本号/HEAD^>` 移动或者回退到相应的版本，前提是没有提交到远程库



## 远程仓库

- 学会了如何创建自己的远程仓库并和自己本次仓库关联
- 学会如何将github上的仓库克隆到本地



## 分支管理

#### 创建和合并分支

有一个主分支master分支，`master`分支是一条线，Git用`master`指向最新的提交，再用`HEAD`指向`master`，就能确定当前分支，以及当前分支的提交点：

![image-20230408152759164](C:\Users\Kivi\AppData\Roaming\Typora\typora-user-images\image-20230408152759164.png)

创建分支：

使用命令`$ git checkout -b dev`创建一个名为dev分支，并且切换到该分支下



![image-20230408153124894](C:\Users\Kivi\AppData\Roaming\Typora\typora-user-images\image-20230408153124894.png)

合并分支：

使用命令`$ git merge dev`把`dev`分支的工作成果合并到`master`分支上

切换回`master`分支：

```
$ git checkout master
```

### 小结

Git鼓励大量使用分支：

查看分支：`git branch`

创建分支：`git branch <name>`

切换分支：`git checkout <name>`或者`git switch <name>`

创建+切换分支：`git checkout -b <name>`或者`git switch -c <name>`

合并某分支到当前分支：`git merge <name>`

删除分支：`git branch -d <name>`

合并分支时，加上`--no-ff`参数就可以用普通模式合并，合并后的历史有分支，能看出来曾经做过合并，而`fast forward`合并就看不出来曾经做过合并。

### Bug分支

修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除；

当手头工作没有完成时，先把工作现场`git stash`一下，然后去修复bug，修复后，再`git stash pop`，回到工作现场；

在master分支上修复的bug，想要合并到当前dev分支，可以用`git cherry-pick <commit>`命令，把bug提交的修改“复制”到当前分支，避免重复劳动

### feature分支

开发一个新feature，最好新建一个分支；

如果要丢弃一个没有被合并过的分支，可以通过`git branch -D <name>`强行删除

#### 协作开发

- 查看远程库信息，使用`git remote -v`；
- 本地新建的分支如果不推送到远程，对其他人就是不可见的；
- 从本地推送分支，使用`git push origin branch-name`，如果推送失败，先用`git pull`抓取远程的新提交；
- 在本地创建和远程分支对应的分支，使用`git checkout -b branch-name origin/branch-name`，本地和远程分支的名称最好一致；
- 建立本地分支和远程分支的关联，使用`git branch --set-upstream branch-name origin/branch-name`；
- 从远程抓取分支，使用`git pull`，如果有冲突，要先处理冲突。

rebase：

- rebase操作可以把本地未push的分叉提交历史整理成直线；
- rebase的目的是使得我们在查看历史提交的变化时更容易，因为分叉的提交需要三方对比。



### 标签

1. 首先，切换到需要打标签的分支上
2. 敲命令`git tag <name>`就可以打一个新标签

 <!--注意：标签总是和某个commit挂钩。如果这个commit既出现在master分支，又出现在dev分支，那么在这两个分支上都可以看到这个标签。-->

- 命令`git tag <tagname>`用于新建一个标签，默认为`HEAD`，也可以指定一个commit id；
- 命令`git tag -a <tagname> -m "blablabla..."`可以指定标签信息；
- 命令`git tag`可以查看所有标签。

#### 远程控制标签

- 命令`git push origin <tagname>`可以推送一个本地标签；
- 命令`git push origin --tags`可以推送全部未推送过的本地标签；
- 命令`git tag -d <tagname>`可以删除一个本地标签；
- 命令`git push origin :refs/tags/<tagname>`可以删除一个远程标签。

使用github

fork别人的仓库

克隆到自己本地

修改并推送至github自己的本地仓库

通过pull request将修改提交给官方，官方考虑接收