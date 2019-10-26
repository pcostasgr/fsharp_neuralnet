// Learn more about F# at http://fsharp.org


open System
open BNN.MathMod
open BNN.NeuralNet

[<EntryPoint>]
let main argv =

    printfn "Yeditech Neural Network 10/2019 "

    printfn "-------------------------------------------------------------------------------"
    printfn "Train network"
    //List.iter (fun x -> printfn "elem:%d" x ) result  

    let input={
            m=[|3.0 ;3.0 ; 3.0;3.0 |]
            height=1
            width=4
        }
        
    let unitWeights=identityMtrx 4 2

    let zeroBias={
            m=[|0.0;0.0|]
            height=1
            width=2
        }

    let actFun = fun f -> 2.0*f

    let layer=
            {
                    input=input
                    weights=createRandMtrx 4 2 
                    bias=createRandMtrx 1 2
            }

    let layer2={
            input=identityMtrx 1 2
            weights=createRandMtrx 2 2
            bias=createRandMtrx 1 2 
        } 


    let derivs =(identityMtrx 3 2,identityMtrx 1 2)

    let network=[ (layer,derivs) ; (layer2,derivs) ]
    let actualOutput ={
            m=[|1.0;0.0|]
            height=1
            width=2 
        }

    let inputList=[(input,actualOutput)]

    let trainResult=TrainNN 2 inputList network sigmoid sigmoidDeriv 0.6

    match trainResult with 
    | NodeList n ->
         printfn "New Network"
         let execResult=ExecuteNN input n sigmoid
         match execResult with 
         | NeuralNetResult (newNetwork,output) ->
                            printMtrx3 output "Output:"
         | NeuralNetFailure f -> printfn "exec failure %s" f  
         
    | Failure f  -> printfn "failure %s" f 


    0 // return an integer exit code