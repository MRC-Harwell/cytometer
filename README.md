# Summary

This is project "cytometer" by Ramón Casero working at the Mammalian Genetics 
Unit (MGU) Medical Research Council (MRC) Harwell, UK.

The current goal of this project is to create a deep convolutional neural
network (CNN) to segment adipocytes from histology images.

# Common problems when running scripts

* From the spyder IDE, sometimes loading or running models will produce a 

        Kernel died, restarting
    
    Restarting the kernel will not fix the problem. You need to reset the GPU, 
    by logging out and back in again into Ubuntu.
    
* Sometimes trying to import keras or theano will throw an error saying that 
pygpu is not available. This error can be fixed by deleting the local environment

        conda remove --name DeepCell --all
    
    and then reinstalling again
    
        ./install_dependencies.sh
        
* Spyder IDE hangs when `%matplotlib qt5` and `import theano` or `import keras`
are combined. The solution is to not use `%matplotlib qt5` and instead configure
Spyder with `Tools->Preferences->IPython console->Graphics->Backend = Automatic`.