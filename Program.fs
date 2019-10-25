// Learn more about F# at http://fsharp.org


open System
open BNN.MathMod
open BNN.NeuralNet

 type MyResult<'R,'F> =
        | Res of 'R
        | Fail of 'F


type SomeType=string*string

[<EntryPoint>]
let main argv =

    printfn "Yeditech Neural Network 10/2019"

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
    | NodeList n -> printfn "New Network"
    | Failure f  -> printfn "failure %s" f 

(*
    let result=ExecuteNN input network sigmoid
    
    match result with 
        | NeuralNetResult (newNetwork, m) ->
            printMtrx3 m  "Output"

            printfn "DEBUG 1-------------------------------------------------------" 
            printNeuralNet newNetwork
            printfn "DEBUG 2-------------------------------------------------------" 
            let trainResult=trainNetwork m actualOutput newNetwork 0.6 sigmoidDeriv 

            match trainResult with 
            | NodeList l -> 
                printfn "New network !!!!"
                printNeuralNet l
            | Failure f -> printfn "failure: %s " f  

        | NeuralNetFailure f -> printfn "General fail %s"  f
*)

    0 // return an integer exit code