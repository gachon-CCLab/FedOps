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
