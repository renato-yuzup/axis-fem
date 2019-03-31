# axis-fem
Hybrid CPU-GPU Finite Element Software for Structural Analysis in Mechanical Engineering

## Preliminary notice ##
** As I'm really busy, I couldn't complete tidying up code dependencies. Also, as I'm in the middle of making changes do support Unix-like systems and automate build process, don't expect to have a beautiful FEM solver by the end of this Readme. Code was fast published to foster research and engage curious people :P In the meantime you can checkout features discussed in my thesis.  **

## What is this? ##
This is the codebase for a finite element solver for structural mechanics that I developed in my doctorate thesis available  [here](http://www.teses.usp.br/teses/disponiveis/3/3151/tde-24122014-120113/pt-br.php) (please note: in Portuguese). I fast published the codebase so that everyone could explore a little bit about my work. But, please, don't expect to have a complete project with automated build and tests and etc because I still lack time to do the whole thing (however rest assured that I'm still actively working on it ;) ).

## Prerequisites ##
- Boost Libraries
- TinyXML

## Examples? ##
- Still working on it, but if you have good talent exploring code, you can check input files in the `Model Input Files` folder to see how data is input to the program and check how the parsers, library manager and solver make everything up (

## How to build? ##
- Still working on it, but the original code was developed in Visual Studio 2010 (and lastly upgraded to 2017 version). VS-project files are included. Because I would infringe copyright adding third-party libraries here, they are not included so you have to download yourself. 

## References ##
YAMASSAKI, Renato T. (2014) An object-oriented finite element program in GPU for nonlinear dynamic structural analysis. Doctorate thesis, University of São Paulo, Brazil.
DOI: [10.11606/T.3.2014.tde-24122014-120113](https://www.doi.org/10.11606/T.3.2014.tde-24122014-120113)
