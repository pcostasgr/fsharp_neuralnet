namespace BNN
open MathMod

module NeuralNet=
   
    type NetLayer={ input:Matrix; weights:Matrix; bias:Matrix;}
    type NetDerivs=Matrix*Matrix 
    type NetNode=NetLayer*NetDerivs

    type NeuralNetErrorType=
        | NeuralNetFailure of string 
        | NodeListError of string 

    type DerivType<'M,'F>=
        |  NodeList of NetNode list
        |  Failure of string  

    type NeuralNetOutput<'M,'F>=
        | NeuralNetResult of NetNode list*Matrix
        | NeuralNetFailure of string 

    let sigmoid=fun x -> 
        let epsilon=exp -x 
        1.0 / ( 1.0 + epsilon )

    let sigmoidDeriv x:float=
        let epsilon=exp -x
        epsilon/ ( (1.0 + epsilon) ** 2.0 )
         
    let executeLayer=fun  (n:NetLayer) (activationFun:float->float)->
        let bind1=mtrxBind ( addMtrx n.bias)
        let bind2=mtrxBind (applyFuncToMtrx activationFun )

        //printMtrx3 n.input "ex input"
        //printMtrx3 n.weights "ex weights"
        //printMtrx3 n.bias "ex bias"
        let nodeOutput= 
            n.weights
            |> mulMtrx n.input 
            |> bind1
            |> bind2
        nodeOutput

    let updateParameters  (m:Matrix) (learningRate:float) (derivative:Matrix) =
        let deriv=mulValueToMtrx derivative learningRate
        let bind1=mtrxBind (subMtrx  m)
        deriv |> bind1
        
    let rec updateNetworkParameters network learningRate =
        match network with 
            | [] ->
                printfn "Empty!!!"
                []
            | hh::tt ->
                let derivs=snd(hh)
                let dW=fst(derivs)
                let dB=snd(derivs)

                printMtrx3 dW "DW"
                printMtrx3 dB "DB"
                let node=fst(hh)
                let weights=node.weights
                let bias=node.bias 

                printMtrx3 weights "weights"
                printMtrx3 bias "bias"
                let newWeights=updateParameters weights learningRate dW
                let newBias=updateParameters bias learningRate dB

                printMtrxS newWeights "new weight "
                printMtrxS newBias "new bias"
                let result =match newWeights with 
                            | MatRes wm -> match newBias with 
                                            | MatRes bm ->                                    
                                                let newNode=({input=node.input;weights=wm;bias=bm},derivs )
                                                [newNode] @ updateNetworkParameters tt learningRate 
                                            | Fail f -> []
                            | Fail f -> []
                result


    let rec trainNetworkDeriv (calcNetwork:NetNode list) (network: NetNode list) (activationDerivFunction:float->float):DerivType<NetNode list,NeuralNetErrorType> =

        printfn "1.trainNetworkDeriv "
        if List.isEmpty network then
            NodeList network
        else 


            if not (List.isEmpty calcNetwork) then
                let prevNode=List.last calcNetwork 
                let size=network.Length-calcNetwork.Length-1

                printfn "size:%d" size 

                let curNode=network.Item size

                let prevdEdB=snd(snd(prevNode))
                let prevW=fst(prevNode).weights

                printMtrx3 prevdEdB "prevdEdB"
                printMtrx3 prevW "prevW"


                let transPrevW=transMtrx prevW 

                //printfn "transPrevW"
                //printMtrx transPrevW

                let bind1=MulMatrix (MatRes prevdEdB) transPrevW
                let layer=fst(curNode)

                let execLayer=executeLayer layer activationDerivFunction

              // printfn "bind1:"
              //  printMtrx bind1

             //   printfn "layer:"
              //  printMtrx execLayer 

                let dEdB=MulVectors bind1 execLayer 
                
                let dEdW=
                    match dEdB with 
                    | MatRes m -> 
                        let transInput=transMtrx layer.input
                        MulMatrix transInput (MatRes m)
                    | Fail f -> Fail (f + " calc dEdB ")

                let newDerivs=match dEdB with
                                | MatRes mb -> 
                                    match dEdW with 
                                    | MatRes mw ->
                                        let nodeDerivs=NetDerivs(mw,mb)
                                        let newNetwork=[NetNode(layer,nodeDerivs)] @ calcNetwork

                                        if newNetwork.Length < network.Length then
                                            trainNetworkDeriv newNetwork network activationDerivFunction
                                        else
                                           NodeList  newNetwork

                                    | Fail f -> Failure  (f + " match newDerivs dEdW " ) 
                                | Fail f-> Failure (f + " Match newDerivs dEdB ")
                
                newDerivs 
            else
                Failure  "Calculated Network must not be empty"


    let trainNetwork output actualOutput network learningRate activationDerivFunction:DerivType<NetNode list,NeuralNetErrorType> =
        let node=List.last network
        let layer=fst(node)
        let derivs=snd(node)

        let outputDiff=subMtrx output actualOutput
        let layerResult=executeLayer layer activationDerivFunction
        let dB=MulVectors outputDiff layerResult
        let dW=MulMatrix (transMtrx layer.input) dB
        match dB with 
        | MatRes mb ->
            match dW with  
            | MatRes mw->
                let calcNetwork=[(layer, (mw,mb))]

                let result=trainNetworkDeriv calcNetwork network activationDerivFunction

                match result with 
                    | NodeList n ->
                        printfn "Update Parameters !!" 
                        let newNetwork=updateNetworkParameters n learningRate 
                        match newNetwork with 
                        | hh::tt -> NodeList (hh::tt)
                        | [] -> Failure ( "Error in updating network weight/bias parameters with derivatives")
                    | Failure f -> Failure (  " Network Fail" + f  )
            | Fail f -> Failure (f + " Weight Derivative Fail Training Network")
        | Fail f -> Failure (f + " Bias Derivative Fail Training Network") 


    let ExecuteNN (input:Matrix) (nn:NetNode list) (activationFunction:float->float ) =
        
        let foldFun acc elem =
            match acc with 
            | NeuralNetResult (netlist,input) ->
                let layer=fst(elem)
                let newLayer={input=input;weights=layer.weights;bias=layer.bias }
                let output=executeLayer newLayer activationFunction
                match output with
                | MatRes m ->NeuralNetResult (nn @ [(newLayer,snd(elem))],m)
                | Fail f -> NeuralNetFailure ( "match output " + f )
            | NeuralNetFailure f -> NeuralNetFailure f 

        let startState=([List.head(nn)],input) 
        List.fold foldFun (NeuralNetResult startState) nn

    let rec TrainNN iterations actualOutput inputs network actFun actDerivFun learnRate=

        let closureFoldFun neuralNet inputNode =
            match neuralNet with 
            | NodeList nn ->
                let calcOutput=ExecuteNN inputNode nn actFun
                match calcOutput with 
                    | NeuralNetResult (netList,output) ->
                        printMtrx3 output "NN Output:"
                        let trainResult=trainNetwork output actualOutput nn learnRate sigmoidDeriv 
                        match trainResult with 
                            | NodeList l -> NodeList l
                            | Failure f -> Failure  ("failure:" + f)  

                    | NeuralNetFailure f -> Failure ("General fail:" + f)
            | Failure f -> Failure f 

        if iterations>0 then

            let newNetwork=List.fold closureFoldFun (NodeList network) inputs 
            let newnn=match newNetwork with 
                        | NodeList nn -> TrainNN (iterations-1)  actualOutput inputs nn actFun actDerivFun learnRate
                        | Failure f -> Failure f 
            newnn
        else
           NodeList network  



    let testCalcOutput=
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

        let result=ExecuteNN input network actFun

        
        let actualOutput ={
            m=[|1.0;0.0|]
            height=1
            width=2 
        }

        match result with 
        | NeuralNetResult (list, m) ->
            printMtrx3 m  "Output"
            let trainResult=trainNetwork m actualOutput network 0.6 sigmoidDeriv 

            match trainResult with 
            | NodeList l -> printfn "New network !!!!"
            | Failure f -> printfn "failure: %s " f  

        | NeuralNetFailure f -> printfn "General fail %s"  f

        0