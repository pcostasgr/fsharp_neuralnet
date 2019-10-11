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

    printfn "Neural Network fsharp 10/2019"
    let layer={
        input={
                m=[|1.0;2.0;3.0;|]
                height=1
                width=3
            }
        weights={
                m=[| 4.0;5.0;2.0;3.0;1.0;5.0|]
                height=3
                width=2
        }
        bias={ 
                m=[| 1.0 ;2.0|]
                height=1
                width=2
        }
    }

    let result=BNN.NeuralNet.executeLayer layer sigmoid


    let nnOutput={
        m=[| 1.0 ; 1.0 |]
        height=1
        width=2 
    }

    let nnDesiredOutput={
        m=[| 1.0 ; 0.0 |]
        height=1
        width=2 
    }

    let nodes=Array.create 1 layer 

    printfn "-------------------------------------------------------------------------------"
    printfn "Train network"
    //List.iter (fun x -> printfn "elem:%d" x ) result  
    let r=testCalcOutput
   // MathMod.printMtrx3 matrix5 "4x4 "|> ignore
    //MathMod.printMtrx matrix4|> ignore
    0 // return an integer exit code