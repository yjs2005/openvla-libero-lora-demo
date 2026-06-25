# Environment Report

- Generated at: 2026-06-26T01:26:30
- Project root: `D:\42系保研准备\openvla-libero-lora-demo`

## System

| Item | Value |
| --- | --- |
| OS | Windows 11 (10.0.26200) |
| Python version | 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)] |
| CPU | 13th Gen Intel(R) Core(TM) i5-13500HX |
| CPU cores / logical CPUs | 20 logical CPUs |
| RAM total | 15.73 GB |
| RAM available | 4.65 GB |
| Disk total | 160.17 GB |
| Disk free | 52.12 GB |

## CLI Tools

| Item | Value |
| --- | --- |
| git available | True |
| git version | git version 2.53.0.windows.1 |
| gh available | False |
| gh version | Not found |

## NVIDIA GPU

| Item | Value |
| --- | --- |
| nvidia-smi present | True |
| nvidia-smi query status | OK |
| CUDA version reported by nvidia-smi | 12.4 |

| GPU | Name | Memory total | Memory used | Memory free | Driver version |
| --- | --- | --- | --- | --- | --- |
| 0 | NVIDIA GeForce RTX 4060 Laptop GPU | 8188 MiB (8.00 GB) | 266 MiB (0.26 GB) | 7692 MiB (7.51 GB) | 551.76 |

<details>
<summary>Raw nvidia-smi output</summary>

```text
Fri Jun 26 01:26:25 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 551.76                 Driver Version: 551.76         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   57C    P0             21W /   87W |     266MiB /   8188MiB |     13%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2092    C+G   ....0_x64__2p2nqsd0c76g0\app\Codex.exe      N/A      |
|    0   N/A  N/A      2832    C+G   D:\FlClash\FlClash.exe                      N/A      |
|    0   N/A  N/A      8260    C+G   ...64__v826wp6bftszj\TranslucentTB.exe      N/A      |
|    0   N/A  N/A     11864    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A     14004    C+G   ...7\extracted\runtime\WeChatAppEx.exe      N/A      |
|    0   N/A  N/A     15832    C+G   ...cw5n1h2txyewy\CrossDeviceResume.exe      N/A      |
|    0   N/A  N/A     16940    C+G   C:\Windows\System32\ShellHost.exe           N/A      |
|    0   N/A  N/A     17176    C+G   ...nt.CBS_cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A     17184    C+G   ...2txyewy\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A     18828    C+G   ...on\149.0.4022.80\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     22536    C+G   ...nt9dgb7efx6bt\app\PredatorSense.exe      N/A      |
|    0   N/A  N/A     22800    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A     23752    C+G   ...iles\Oray\AweSun\flutter\AweSun.exe      N/A      |
|    0   N/A  N/A     28068    C+G   ...325.0_x64__dt26b99r8h8gj\RtkUWP.exe      N/A      |
|    0   N/A  N/A     28948    C+G   ...rkmn4z8aw4\AcerPurifiedVoiceApp.exe      N/A      |
|    0   N/A  N/A     33104    C+G   ....0_x64__2p2nqsd0c76g0\app\Codex.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

</details>

## PyTorch

| Item | Value |
| --- | --- |
| PyTorch installed | Yes |
| torch version | 2.12.0+cpu |
| torch.cuda.is_available() | False |
| torch.version.cuda | None |
| torch.cuda.get_device_name() | N/A |
| torch.cuda.mem_get_info() | N/A |
| PyTorch detection error | None |

## Suitability Assessment

- GPU memory is below 16GB: not recommended for OpenVLA-7B; at most use this machine for environment preparation and very light code tests.
- Disk free space is 30GB to 100GB: can try small LIBERO data and checkpoints, but not suitable for BridgeData V2.
- OpenVLA/LIBERO inference evaluation: Not recommended for OpenVLA-7B on this GPU memory size.
- Small-scale LoRA fine-tuning: Not recommended on this GPU memory size.
- Full large-scale training: Not suitable.
- PyTorch CUDA is not currently available, so GPU workloads may require installing a CUDA-enabled PyTorch build before running OpenVLA.
