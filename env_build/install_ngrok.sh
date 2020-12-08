# ngrok own the port 20000-21000
domain=
make_server=
make_client=
git clone https://github.com/inconshreveable/ngrok.git
cd ngrok
export NGROK_DOMAIN=$domain
openssl genrsa -out rootCA.key 2048
openssl req -x509 -new -nodes -key rootCA.key -subj "/CN=$NGROK_DOMAIN" -days 5000 -out rootCA.pem
openssl genrsa -out device.key 2048
openssl req -new -key device.key -subj "/CN=$NGROK_DOMAIN" -out device.csr
openssl x509 -req -in device.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out device.crt -days 5000

cp rootCA.pem assets/client/tls/ngrokroot.crt
cp device.crt assets/server/tls/snakeoil.crt
cp device.key assets/server/tls/snakeoil.key


make clean

if make_server:
then
    GOOS=linux GOARCH=amd64 make release-server
    cp ./bin/ngrokd /usr/local/bin
    # generate the ngrok server
    echo \
    "[Unit]
Description=ngrokd
After=network.target
[Service]
Type=simple
ExecStart=/usr/loadl/bin/ngrokd -domain="$domain" --httpAddr=":801" -httpsAddr=":802"
[Install]
WantedBy=multi-user.target" > /etc/systemd/system/ngrokd.service
fi

if make_client:
then
    GOOS=linux GOARCH=amd64 make release-client
    cp ./bin/ngrok /usr/local/bin
    echo \
    "server_addr: "ziyan.xyz:4443"
trust_host_root_certs: false
tunnels:
  ssh:
    remote_port: 20001
    proto:
      tcp: 22
" > /usr/local/etc/ngrok.conf
    # generate the ngrok client
    echo \
    "[Unit]
Description=ngrok
After=network.target
[Service]
Type=simple
ExecStart=/usr/loadl/bin/ngrok -config=/usr/loadl/etc/ngrok.conf start ssh #pan-ssh pi-ssh mac-ssh pi-vnc pan-http pi-http bt-http test-http
[Install]
WantedBy=multi-user.target" > /etc/systemd/system/ngrok.service
fi
#    echo \
#    "[Unit]
#Description=ngrok
#After=network.target
#[Service]
#Type=simple
#ExecStart=/usr/loadl/bin/ngrok -config=/usr/loadl/etc/ngrok.conf start ssh #pan-ssh pi-ssh mac-ssh pi-vnc pan-http pi-http bt-http test-http
#[Install]
#WantedBy=multi-user.target" > a.server