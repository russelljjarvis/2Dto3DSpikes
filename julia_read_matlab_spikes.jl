using MAT
vars = matread("105l001p16.mat")
vars["neuralData"]["spikeRasters"]
for i in vars["neuralData"]#["spikeRasters"]
   @show(i)
end
@show(vars["neuralData"]["trialId"])
