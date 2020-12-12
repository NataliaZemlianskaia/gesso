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


selection.metrics = function(dataset, result) {
  result_coef = c(result$beta_0_hat, result$beta_G_hat, result$beta_E_hat, result$beta_GxE_hat)
  
  if (sum(result$beta_GxE_hat != 0) == 0) {
    precision_gxe = 1.0
  } else {
    precision_gxe = sum(result$beta_GxE_hat[dataset$index_beta_gxe_non_zero] != 0) / sum(result$beta_GxE_hat != 0)
  }
  if (sum(result$beta_G_hat != 0) == 0) {
    precision_g = 1.0
  } else {
    precision_g = sum(result$beta_G_hat[dataset$index_beta_non_zero] != 0) / sum(result$beta_G_hat != 0)
  }  
  
  return(list(valid_loss=mean((dataset$Y_valid - linear.predictor(dataset$G_valid, dataset$E_valid, result$beta_0_hat, result$beta_G_hat, result$beta_E_hat, result$beta_GxE_hat))^2) / 2,
              b_g_non_zero=sum(result$beta_G_hat != 0),
              b_gxe_non_zero=sum(result$beta_GxE_hat != 0),
              mse=mse_coef(result_coef, dataset),
              mse_beta = sqrt(mean((result$beta_G_hat[dataset$index_beta_non_zero] - dataset$Beta_G[dataset$index_beta_non_zero])^2)),
              mse_beta_GxE = sqrt(mean((result$beta_GxE_hat[dataset$index_beta_gxe_non_zero] - dataset$Beta_GxE[dataset$index_beta_gxe_non_zero])^2)),
              test_loss=mean((dataset$Y_test - linear.predictor(dataset$G_test, dataset$E_test, result$beta_0_hat, result$beta_G_hat, result$beta_E_hat, result$beta_GxE_hat))^2) / 2,
              sensitivity_g = sum(abs(result$beta_G_hat[dataset$index_beta_non_zero]) != 0)/(length(dataset$index_beta_non_zero) + 1e-8),
              specificity_g = sum(abs(result$beta_G_hat[dataset$index_beta_zero]) == 0)/(length(dataset$index_beta_zero) + 1e-8),
              sensitivity_gxe = sum(abs(result$beta_GxE_hat[dataset$index_beta_gxe_non_zero]) != 0)/(length(dataset$index_beta_gxe_non_zero) + 1e-8),
              specificity_gxe = sum(abs(result$beta_GxE_hat[dataset$index_beta_gxe_zero]) == 0)/(length(dataset$index_beta_gxe_zero) + 1e-8),
              
              precision_g = precision_g,
              precision_gxe = precision_gxe,
              
              auc_gxe=getROC_AUC(result$beta_GxE_hat, dataset$Beta_GxE),
              auc_g=getROC_AUC(result$beta_G_hat, dataset$Beta_G)))
}
