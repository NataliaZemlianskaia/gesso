get.AUC = function(estimated_coef, true_coef){
  estimated_coef = abs(estimated_coef)
  estimated_sorted = sort(estimated_coef, decreasing=TRUE, index.return=TRUE)
  val = unlist(estimated_sorted$x)
  idx = unlist(estimated_sorted$ix)  
  roc_y = true_coef[idx];
  stack_y = cumsum(roc_y != 0)/(sum(roc_y != 0) + 1e-5) #sensitivity
  stack_x = cumsum(roc_y == 0)/(sum(roc_y == 0) + 1e-5)
  auc = sum((stack_x[2:length(roc_y)]-stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
  return(auc)
}


selection.metrics = function(true_b_g, true_b_gxe, estimated_b_g, estimated_b_gxe) {
  index_beta_gxe_non_zero = which(true_b_gxe != 0)
  index_beta_non_zero = which(true_b_g != 0)
  index_beta_gxe_zero = which(true_b_gxe == 0)
  index_beta_zero = which(true_b_g == 0)

    
  if (sum(estimated_b_gxe != 0) == 0) {
    precision_gxe = 1.0
  } else {
    precision_gxe = sum(estimated_b_gxe[index_beta_gxe_non_zero] != 0) / sum(estimated_b_gxe != 0)
  }
  if (sum(estimated_b_g != 0) == 0) {
    precision_g = 1.0
  } else {
    precision_g = sum(estimated_b_g[index_beta_non_zero] != 0) / sum(estimated_b_g != 0)
  }  
  
  return(list(b_g_non_zero=sum(estimated_b_g != 0),
              b_gxe_non_zero=sum(estimated_b_gxe != 0),
              mse_b_g = sqrt(mean((estimated_b_g[index_beta_non_zero] - true_b_g[index_beta_non_zero])^2)),
              mse_b_gxe = sqrt(mean((estimated_b_gxe[index_beta_gxe_non_zero] - true_b_gxe[index_beta_gxe_non_zero])^2)),
              
              sensitivity_g = sum(abs(estimated_b_g[index_beta_non_zero]) != 0)/(length(index_beta_non_zero) + 1e-8),
              specificity_g = sum(abs(estimated_b_g[index_beta_zero]) == 0)/(length(index_beta_zero) + 1e-8),
              precision_g = precision_g,
              sensitivity_gxe = sum(abs(estimated_b_gxe[index_beta_gxe_non_zero]) != 0)/(length(index_beta_gxe_non_zero) + 1e-8),
              specificity_gxe = sum(abs(estimated_b_gxe[index_beta_gxe_zero]) == 0)/(length(index_beta_gxe_zero) + 1e-8),
              precision_gxe = precision_gxe,
              
              auc_g=get.AUC(estimated_b_g, true_b_g),
              auc_gxe=get.AUC(estimated_b_gxe, true_b_gxe)))
}
