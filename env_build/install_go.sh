go_version=
wget -c https://golang.google.cn/dl/go$go_version.linux-amd64.tar.gz

sudo tar -zxvf go$go_version.linux-amd64.tar.gz -C ~/.local
echo 'export GOROOT=$HOME/.local/go' >> ~/.bashrc
echo 'export PATH=$PATH:$GOROOT/bin' >> ~/.bashrc