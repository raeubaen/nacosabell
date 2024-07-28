root << EOF
new TTree("t", "t")
t->ReadFile("roc.csv")
new TCanvas("c")
t->Draw("tpr:fpr")
c->SaveAs("roc.root")
