# FedOps: Federated Learning Lifecycle Operations Management Platform
  

<p align="center">
    <a href="http://ccljhub.gachon.ac.kr:40020/">FedOps</a> |
    <a href="https://join.slack.com/t/fedopshq/shared_invite/zt-1xvo9pkm8-drLEdtOT1_vNbcXoxGmQ5A">Slack</a> |
    <a href="https://www.linkedin.com/company/89975476/admin/">LinkedIn</a> |
    <a href="https://sites.google.com/view/keylee/">CCL Site</a> |
    <a href="https://www.youtube.com/watch?v=9Ns0q4zHfLk/">Youtube</a>
    <br /><br />
</p>


[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/gachon-CCLab/FedOps/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/fedopshq/shared_invite/zt-1xvo9pkm8-drLEdtOT1_vNbcXoxGmQ5A)


FedOps (`fedops`) is a platform that helps organizations effectively manage and coordinate their federated learning operations:

* **FLScalize**: It simplifies the application of data and models in a FL environment by leveraging Flower's Client and Server.

* **Manager**: The manager oversees and manages the real-time FL progress of both clients and server

* **CE/CS**: Contribution Evaluation and Client Selection processes based on their performance.

* **CI/CD/CFL**: the CI/CD/CFL system seamlessly integrates with a Code Repo, 
enabling code deployment to multiple clients and servers for continuous or periodic federated learning.


* **Monitoring**: The FL dashboard is available for monitoring and observing the lifecycle of FL clients and server


## FedOps Tutorial

FedOps has developed a web service to manage the lifecycle operations of federated learning on real devices.
* **Install FedOps Library**
```bash
$ pip install fedops
```

### Run FedOps Agent Studio

Docker Desktop 또는 Docker Engine이 설치된 환경에서 다음 명령으로 실행합니다.

```bash
fedops run agent-studio
```

이 명령은 현재 OS와 CPU architecture에 맞는
`gachonccl/fedops-agent-studio:latest` 이미지를 확인하고,
`~/fedops-workspace`를 로컬 Workspace로 연결한 뒤 브라우저를 엽니다.
Task별 Python 환경과 데이터는 Workspace에 유지하고, 반복 환경 Sync에 사용하는 uv
package/Python cache는 `fedops-agent-studio-uv` Docker volume에 보존합니다. 따라서
컨테이너와 이미지가 교체되어도 cache를 재사용하며, 특히 Windows Docker Desktop에서
불필요한 재다운로드와 압축 해제 비용을 줄입니다.
NVIDIA Container Runtime이 준비된 Linux/Windows 환경에서는 GPU를 자동 사용하고,
그 외 환경에서는 CPU 모드로 실행합니다.

기본 Docker bind address는 `0.0.0.0`입니다. 따라서 같은 실행으로
`localhost`, `127.0.0.1`, 호스트의 LAN IP에서 접속할 수 있습니다. 호스트
네트워크에 공개되므로 OS firewall에서 허용 범위를 관리해야 합니다. 로컬 장치에서만
접속하려면 다음 옵션을 사용합니다.

```bash
fedops run agent-studio --bind-address 127.0.0.1
```

실제 컨테이너를 변경하지 않고 실행 구성을 먼저 확인할 수 있습니다.

```bash
fedops run agent-studio --dry-run
```

Workspace, Studio port 또는 GPU 모드를 명시하려면 다음 옵션을 사용합니다.

```bash
fedops run agent-studio --workspace ~/fedops-workspace --port 24368 --gpu auto
```

Agent Studio와 로컬 Host integration을 종료하고 결과를 확인하려면 다음 명령을
사용합니다.

```bash
fedops stop agent-studio
```

폴더 열기만 실패하면 **Docker가 실행 중인 컴퓨터**의 터미널에서 다음 명령으로
호스트 연결을 복구하고 컨테이너 내부에서 연결을 확인합니다. 이미지 다운로드나
Studio 재시작 없이 현재 Workspace와 포트를 사용합니다.

```bash
fedops run agent-studio --repair-host
```

`--repair-host`는 FedOps 패키지 1.1.30.18부터 지원합니다.
이 옵션이 없다면 해당 버전 배포 후 `python -m pip install -U fedops`로 호스트의 FedOps를 업데이트합니다.
Docker 이미지 업데이트와 호스트에 설치된 FedOps 업데이트는 별개입니다.
PC 재부팅 후 Docker에서 컨테이너만 자동 시작됐다면 위 복구 명령이나
`fedops run agent-studio`로 호스트 프로그램도 시작해야 합니다.
원격 브라우저로 접속한 경우 폴더는 브라우저 컴퓨터가 아니라 Docker 호스트에서 열립니다.
연결 시간 초과가 계속되면 해당 호스트의 방화벽/VPN 설정을 확인합니다.
Linux에서 폴더 창을 열려면 그래픽 데스크톱과 `xdg-open`이 필요합니다.

### Real Devices
* [Start FedOps Silo](https://github.com/gachon-CCLab/FedOps/tree/main/silo/examples/torch)
* [Start FedOps Mobile](https://github.com/gachon-CCLab/FedOps/tree/main/mobile/examples)

### Single Machine
* [Start FedOps Simualtion](https://github.com/gachon-CCLab/FedOps/tree/main/simulation/examples)



## Community

<a href="https://github.com/gachon-CCLab/FedOps/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=gachon-CCLab/FedOps" />
</a>

## Paper
<a href="https://ieeexplore.ieee.org/document/10122960">**FLScalize: Federated Learning Lifecycle Management**</a>

```bibtex
@article{Cognitive Computing Lab,
  title={FLScalize: Federated Learning Lifecycle Management},
  author={Semo Yang; Jihwan Moon; Jinsoo Kim; Kwangkee Lee; Kangyoon Lee}, 
  journal={IEEE Access},
  Page(s)={47212 - 47222}
  DOI={10.1109/ACCESS.2023.3275439}
  year={2023}
}
```


## Support
For any questions or issues, please contact the FedOps support team at <U>***gyom1204@gachon.ac.kr***</U>
