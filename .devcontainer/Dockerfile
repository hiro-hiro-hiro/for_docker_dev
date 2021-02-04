# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.154.2/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: bionic, focal
ARG VARIANT="focal"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# ベースイメージ名:タグ名
FROM continuumio/anaconda3:2019.03

# pipをアップグレードし必要なパッケージをインストール
RUN pip install --upgrade pip && \
    pip install autopep8 && \
    pip install Keras && \
    pip install tensorflow 

# Install Julia
#RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz

# Juliaのパッケージインストール
#RUN julia -e 'using Pkg; Pkg.add(["IJulia"]); using IJulia'
#RUN sudo apt-get install -y build-essential
#RUN julia -e 'using Pkg; Pkg.add(["PackageCompiler"]); using PackageCompiler'
#RUN sudo apt-get install -y qt5-default
#RUN julia -e 'using Pkg; Pkg.add(["UnicodePlots", "GR", "Plots"]); \
#              using UnicodePlots, GR, Plots'
#RUN julia -e 'using Pkg; Pkg.add(["Distributions"]); using Distributions'
#RUN julia -e 'using Pkg; Pkg.add(["LinearAlgebra"]); using LinearAlgebra'
#RUN julia -e 'using Pkg; Pkg.add(["PDMats"]); using PDMats'
#RUN julia -e 'using Pkg; Pkg.add(["ProgressBars"]); using ProgressBars; '

# コンテナ側のルート直下にworkdir/（任意）という名前の作業ディレクトリを作り移動する
WORKDIR /workdir

# コンテナ側のリッスンポート番号
# 明示しているだけで、なくても動く
EXPOSE 8888

# ENTRYPOINT命令はコンテナ起動時に実行するコマンドを指定（基本docker runの時に上書きしないもの）
# "jupyter-lab" => jupyter-lab立ち上げコマンド
# "--ip=0.0.0.0" => ip制限なし
# "--port=8888" => EXPOSE命令で書いたポート番号と合わせる
# ”--no-browser” => ブラウザを立ち上げない。コンテナ側にはブラウザがないので 。
# "--allow-root" => rootユーザーの許可。セキュリティ的には良くないので、自分で使うときだけ。
# "--NotebookApp.token=''" => トークンなしで起動許可。これもセキュリティ的には良くない。
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# CMD命令はコンテナ起動時に実行するコマンドを指定（docker runの時に上書きする可能性のあるもの）
# "--notebook-dir=/workdir" => Jupyter Labのルートとなるディレクトリを指定
CMD ["--notebook-dir=/workdir"]