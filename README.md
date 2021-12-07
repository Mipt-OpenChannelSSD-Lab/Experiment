# How to configure QEMU Open-Channel SSD 2.0

1) Download all required packages for linux from here https://wiki.qemu.org/Hosts/Linux
Without first line nothing will work, other packages I've also installed

```
sudo apt-get install git libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev

sudo apt-get install git-email
sudo apt-get install libaio-dev libbluetooth-dev libbrlapi-dev libbz2-dev
sudo apt-get install libcap-dev libcap-ng-dev libcurl4-gnutls-dev libgtk-3-dev
sudo apt-get install libibverbs-dev libjpeg8-dev libncurses5-dev libnuma-dev
sudo apt-get install librbd-dev librdmacm-dev
sudo apt-get install libsasl2-dev libsdl1.2-dev libseccomp-dev libsnappy-dev libssh2-1-dev
sudo apt-get install libvde-dev libvdeplug-dev libvte-2.90-dev libxen-dev liblzo2-dev
sudo apt-get install valgrind xfslibs-dev 

sudo apt-get install libnfs-dev libiscsi-dev
```

2) Install KVM https://losst.ru/ustanovka-kvm-ubuntu-16-04#Установка_KVM_в_Ubuntu_2004
Do everything described in section "УСТАНОВКА KVM В UBUNTU 20.04". If you have another linux dictributive, search for a manual yourself)

3) Install QEMU itself 
If you encounter error while configuring QEMU, add --disable-werror to bypass it (I've added) or try to fix, if you want
To avoid problems, I've installed everything with root priviledges
```
git clone --recurse-submodules https://github.com/OpenChannelSSD/qemu-nvme.git
cd qemu-nvme
mkdir build && cd build
../configure --target-list=x86_64-softmmu --enable-debug
make -jN
make -jN install
```

4) Configure device. Full instruction here https://github.com/OpenChannelSSD/qemu-nvme
You only need to do the following
```
qemu-img create -f ocssd -o num_grp=2,num_pu=4,num_chk=60 ocssd.img
```
5) Download linux .iso file. For example, ubuntu https://ubuntu.com/download/desktop
6) To run VM without VNC (it's simpilier, IMHO) install this UI packages
```
sudo apt-get install gtk2.0
sudo apt-get install build-essential libgtk2.0-dev
```
7) Run VM
```
x86_64-softmmu/qemu-system-x86_64 -m 1024 -enable-kvm \
-blockdev ocssd,node-name=nvme01,file.driver=file,file.filename=ocssd.img \
-device nvme,drive=nvme01,serial=deadbeef,id=lnvm \
-cdrom <path to .iso file>
```

If you face troubles with audio driver, dissable it by passing env variable `QEMU_AUDIO_DRV=none` to launch command

8) Try this command https://openchannelssd.readthedocs.io/en/latest/gettingstarted/
or this http://lightnvm.io/pblk-tools/usage.html

I've encountered troubles with this command `sudo nvme lnvm create -d nvme0n1 --lun-begin=0 --lun-end=3 -n mydevice -t pblk`

9) More information can be found here http://lightnvm.io/ . More documentation also search here

As I understood, developers aren't maintaining ocssd project now. They are focused on this https://zonedstorage.io/ It looks very similar to what we want, but should be explored...