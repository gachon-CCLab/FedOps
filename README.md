# FedOps: Federated Learning Lifecycle Operations Management Platform

<p align="center">
  <a href="https://flower.dev/">
    <img src="https://flower.dev/_next/image/?url=%2F_next%2Fstatic%2Fmedia%2Fflower_white_border.c2012e70.png&w=640&q=75" width="140px" alt="Flower Website" />
  </a>
</p>
<p align="center">
    <a href="https://sites.google.com/view/keylee/">CCL Site</a> |
    <a href="https://www.linkedin.com/company/89975476/admin/">LinkedIn</a>
    <br /><br />
</p>

FedOps (`fedops`) is a platform that helps organizations effectively manage and coordinate their federated learning operations:

* **FLScalize**: It simplifies the application of data and models in a FL environment by leveraging Flower's Client and Server.

* **Manager**: The manager oversees and manages the real-time FL progress of both clients and server

* **CE/CS & BCFL**: Contribution Evaluation and Client Selection processes incentivize individual clients through a BCFL function based on their performance.


* **CI/CD/CFL**: the CI/CD/CFL system seamlessly integrates with a Code Repo, 
enabling code deployment to multiple clients and servers for continuous or periodic federated learning.


* **Monitoring**: The FL dashboard is available for monitoring and observing the lifecycle of FL clients and server


## FedOps on Real Device Tutorial

FedOps has developed a web service to manage the lifecycle operations of federated learning on real devices.

* [Start (TensorFlow)](https://github.com/gachon-CCLab/FedOps/tree/main/real_device/example)



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