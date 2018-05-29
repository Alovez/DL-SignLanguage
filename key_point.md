1. Validate the Dateset manually.
1. Do some transform before put the date to CNN, so your AI can get what they need easier
1. In `torch==0.4.0`, you should add `.cuda()` when init the module to enable GPU
1. When you use the module as `.cuda()`, you should change all the date to `cuda` too
1. `nn.Couv2d()` require a 4D Tensor, so you may need resize your picture to 4D first
1. You can `print(youe_module)` directly to see the structure.
