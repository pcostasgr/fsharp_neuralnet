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

    let input1={
            m=[|3.0 ;3.0 ; 0.0;0.0 |]
            height=1
            width=4
        }

    let input2={
            m=[|0.0 ;0.0 ; 3.0;3.0 |]
            height=1
            width=4
        }

    let input3={
            m=[|0.0 ;0.0 ; 3.0;3.0 |]
            height=1
            width=4
        }

    let unitWeights=identityMtrx 4 2

    let zeroBias={
            m=[|0.0;0.0|]
            height=1
            width=2
        }

    let layer=
            {
                    input=identityMtrx 1 4
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

    let output1 ={
            m=[|1.0;0.0|]
            height=1
            width=2 
        }

    let output2 ={
            m=[|0.0;1.0|]
            height=1
            width=2 
        }

    let inputList=[ (input1,output1) ; (input2,output2)  ]

    let timer=System.Diagnostics.Stopwatch()

    let iterations=1000
    let learningRate=0.4
    printfn "Start training"
    printfn "Layer no:%i" network.Length
    printfn "Iterations:%i" iterations 
    printfn "Learning Rate:%f" learningRate

    timer.Start()
    let trainResult=TrainNN iterations inputList network sigmoid sigmoidDeriv learningRate

    match trainResult with 
    | NodeList n ->
         printfn "New Network"
         let execResult=ExecuteNN input3 n sigmoid
         match execResult with 
         | NeuralNetResult (newNetwork,output) ->
                            printMtrx3 output "Final Output:"
         | NeuralNetFailure f -> printfn "exec failure %s" f  
         
    | Failure f  -> printfn "General failure %s" f 

    let stopValue=timer.ElapsedMilliseconds
    printfn "Elapsed time:%i ms" stopValue
    0 // return an integer exit code