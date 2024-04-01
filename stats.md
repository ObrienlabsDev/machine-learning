
## Running tflow.py with 25 epochs and 4096 batch size
### AD102 NVidia 4090 Ada Dual 384bit 1008GB/s 24G+24G PCIe 8x only - 32768 cores
98ms 70c 270x2w 91-70%

### AD102 NVidia 4090 Ada 24G PCIe 8x - 16384 cores
137ms 66c 360w 97-93%

### GA102 NVidia A6000 Ampere 384bit 768GB/s 48G PCIe 16x - 10752 cores
- The motherboard will affect the performance of an A6000 with up to 30% performance degradation - making an A6000 behave like an A4500
- 390ms 100c tt 308w 100-90% on 790 maximus with full power availablity 1600w psu
- 510ms 92c 280w 92-70% - on 790 hero with lower power availability 1500w psu
### GA102 NVidia A4500 Ampere Dual 320bit 640GB/s 40G NVLink - 14336 cores
381ms 90c 400w 97-57%

### GA102 NVidia A4500 Ampere 320bit 640GB/s 20G PCIe 8x - 7168 cores
511ms 90c 205w 99-92%
524ms 90c 209w 98-90%

### GA102 NVidia A4500 Ampere 320bit 640GB/s 20G PCIe 16x - 7168 cores
505ms 90c 205w 99-96%

### GA104 NVidia A4000 Ampere 256bit 448GB/s 16G PCIe 16x - 6144 cores
778ms 92c 140w 92-70%

### AD104 NVidia 3500 Ada Mobile 192bit 432GB/s 12G - 5120 cores
552ms 110c 96w 100%

### TU104 NVidia RTX-5000 Mobile 256bit 448GB/s 16G PCIe 3.0 16x - 3072 cores
990ms 118w 99-92%
### GP107 NVidia P1000 Pascal 4G - cores
