namespace BNN


module MathMod=

    type Matrix={m:float [];height:int;width:int;}

    type MatrixResult<'M,'F>=
        | MatRes of Matrix
        | Fail of string

   
    
    let initMtrx=
        {
            m=Array.empty
            height=0
            width=0
        }

    let identityMtrx=fun (h:int) (w:int) ->
        let size=h*w 

        {
            m=[|
                for y=0 to h-1 do
                    for x=0 to w-1 do
                        if y*w+x<>y*w+y then
                            yield 0.0
                        else
                            yield 1.0 
              |]
            height=h
            width=w 
        }


    let printMtrx2=fun (ma:Matrix)->
        printfn "-----------------------------------------"
        printfn "%d x %d  col rows" ma.height ma.width
        for i=0 to ma.height-1 do
            for j=0 to ma.width-1 do
                printf "%f " ma.m.[i*ma.width+j]
            printfn ""
        printfn ""
        None |> ignore 
    
    let printMtrx3=fun (ma:Matrix) (msg:string)->
        printfn "-----------------------------------------"
        printfn "%s" msg 
        printMtrx2 ma

    let printMtrx =fun m ->
        match m with
        | MatRes m ->printMtrx2 m
        | Fail f -> printfn "%s" f 
                    None  |> ignore 
    let printMtrxS =fun m s ->
        match m with
        | MatRes m ->printMtrx3 m s
        | Fail f -> printfn "%s" f 
                    None  |> ignore 


    let addMtrx=fun (ma:Matrix) (mb:Matrix) -> 
        if ma.height=mb.height && ma.width=mb.width then
           MatRes {
                m=Array.map2 (+) ma.m mb.m 
                height=ma.height
                width=ma.width
            }
        else
           let msg= sprintf "addMtrx Matrices do not have the same size heigh1:%i height2:%i width1:%i width2:%i " ma.height mb.height ma.width mb.width
           Fail msg 
            
    let subMtrx=fun (ma:Matrix) (mb:Matrix) -> 
        if ma.height=mb.height && ma.width=mb.width then
           MatRes {
                m=Array.map2 (-) ma.m mb.m 
                height=ma.height
                width=ma.width
            }
        else
           let msg= sprintf "subMtrx Matrices do not have the same size heigh1:%i height2:%i width1:%i width2:%i " ma.height mb.height ma.width mb.width
           Fail msg 


    let mulValueToMtrx =fun (ma:Matrix) (v:float) ->
        MatRes {
            m=Array.map(fun x -> x*v) ma.m
            height=ma.height
            width=ma.width 
        }

    let applyFuncToMtrx =fun (f:float->float) (ma:Matrix) -> 
        MatRes {
            m=Array.map f ma.m
            height=ma.height
            width=ma.width
        }


    let transMtrx=fun (ma:Matrix) ->
        MatRes {
            m=[|
                for w=0 to ma.width-1 do
                    for h=0 to ma.height-1 do
                        yield ma.m.[h*ma.width+w]
            |]
            height=ma.width
            width=ma.height
        }


    let mulMtrx=fun (ma:Matrix) (mb:Matrix) ->
        if ma.width=mb.height then
            MatRes {
                m=[|
                    for y=0 to ma.height-1 do
                        for x=0 to mb.width-1 do
                           yield  [|
                                for z=0 to ma.width-1 do
                                    yield ma.m.[y*ma.width+z]*mb.m.[z*mb.width+x]                                    
                            |] |> Array.sum

                |]
                height=ma.height
                width=mb.width
            }
        else
            let msg=sprintf "mulMtrx error matrices have different dimensions ma width1:%d mb height2:%d " ma.width mb.height       
            Fail msg


    let mulVector (ma:Matrix) (mb:Matrix)=
        if ma.height=1 && mb.height=1 && ma.width=mb.width then
            MatRes {
                m=Array.map2 (fun x y -> x*y) ma.m mb.m
                height=1
                width=ma.width 
            }
        else
            let msg=sprintf "mulVector error matrices have different dimensions  width1:%d width2:%d  height1:%d height2:%d" 
                        ma.width 
                        mb.width
                        ma.height
                        mb.height       
            Fail msg 

    
    let randomArray01=fun (size:int) -> 
         let rand=System.Random()
         [| 
             for i=0 to size-1 do 
               yield rand.NextDouble()-0.5
         |]

    let createRandMtrx=fun (height:int) (width:int) ->
        let size=height*width
        {
            m=randomArray01 size
            height=height
            width=width 
        }

    let mtrxBind =fun f m  ->
        match m with 
        | MatRes m -> f m
        | Fail f    -> Fail f 
        
    let MulMatrix a b =
                match a with 
                    | MatRes m1 ->
                        match b with
                        | MatRes m2 -> mulMtrx m1 m2  
                        | Fail f -> Fail f 
                    | Fail f -> Fail f 

    let MulVectors a b =
                match a with 
                    | MatRes m1 ->
                        match b with
                        | MatRes m2 -> mulVector m1 m2  
                        | Fail f -> Fail f 
                    | Fail f -> Fail f 


