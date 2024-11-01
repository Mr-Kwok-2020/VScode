pip install 

conda activate py39

# 创建虚拟环境
conda create --name py39 python=3.9 

# 激活虚拟环境
conda activate py39
conda activate py39_tf 

# 退出虚拟环境
conda deactivate  

# 列出所有虚拟环境
conda env list  

# 在虚拟环境中安装包
conda install package_name  

# 列出当前虚拟环境中安装的所有包
conda list  

# 导出当前虚拟环境的配置到一个 YML 文件中


# 从虚拟环境中卸载包
conda remove package_name  

# 从 YML 文件创建虚拟环境
conda env create -f environment.yml  

# 更新 Anaconda 到最新版本
conda update anaconda     

# 删除一个Conda环境：
conda env remove --name myenv
conda env remove --name py_GT_gan
