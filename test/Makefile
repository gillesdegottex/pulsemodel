# Prepare or clean data for tests
#
# Usage
# 	$ make
#
# Copyright(C) 2016 Engineering Department, University of Cambridge, UK.
#
# License
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Author
#     Gilles Degottex <gad27@cam.ac.uk>

all:
	$(MAKE) samples.scp

samples.scp:
	wget http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0010.wav -O slt_arctic_a0010.wav
	wget http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0020.wav -O bdl_arctic_a0020.wav
	wget http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_clb_arctic/wav/arctic_a0030.wav -O clb_arctic_a0030.wav
	wget http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_awb_arctic/wav/arctic_a0040.wav -O awb_arctic_a0040.wav
	ls *.wav |grep -v '.resynth.wav' > samples.scp

clean:
	rm -f *.resynth.wav *.f0 *.lf0 *.logf0 *.spec *.fwspec *.fwcep *.pdd *.mpdd *.nm *.bndnm *.fwnm *.diff

distclean:
	rm -f samples.scp *.wav *.f0 *.lf0 *.f0txt *.logf0 *.spec* *.fwspec *.fwlspec *.fwcep *.pdd *.mpdd *.nm *.bndnm *.fwnm *.diff
