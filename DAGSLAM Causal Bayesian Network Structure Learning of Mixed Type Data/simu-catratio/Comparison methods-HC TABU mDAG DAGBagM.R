# Define accuracy metrics function ----------------------  
count_accuracy <- function(B_true, B_est) {  
  # Validate function arguments  
  if (!all(B_est %in% c(0, 1))) {  
    stop("B_est should take value in {0,1}")  
  }  
  
  d <- nrow(B_true)  
  # Linear index of non-zero elements  
  pred <- which(B_est != 0)  
  cond <- which(B_true != 0)  
  cond_reversed <- which(t(B_true) != 0)  
  cond_skeleton <- unique(c(cond, cond_reversed))  
  
  # True positives, false positives, and reversals  
  true_pos <- intersect(pred, cond)  
  false_pos <- setdiff(pred, cond_skeleton)  
  extra <- setdiff(pred, cond)  
  reverse <- intersect(extra, cond_reversed)  
  
  # Compute accuracy metrics  
  pred_size <- length(pred)  
  cond_neg_size <- 0.5 * d * (d - 1) - length(cond)  
  fdr <- (length(reverse) + length(false_pos)) / max(pred_size, 1)  
  tpr <- length(true_pos) / max(length(cond), 1)  
  fpr <- (length(reverse) + length(false_pos)) / max(cond_neg_size, 1)  
  
  # Get lower triangle matrix  
  get_lower_triangle <- function(mat) {  
    if (!is.matrix(mat)) {  
      stop("Input must be a matrix.")  
    }  
    
    # Create a matrix with the same dimensions as the input matrix, initialized to zero  
    lower_triangle <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))  
    
    # Copy the lower triangle values to the new matrix  
    lower_triangle[lower.tri(mat, diag = TRUE)] <- mat[lower.tri(mat, diag = TRUE)]  
    
    return(lower_triangle)  
  }  
  
  pred_lower <- which(get_lower_triangle(B_est + t(B_est)) != 0)  
  cond_lower <- which(get_lower_triangle(B_true + t(B_true)) != 0)  
  extra_lower <- setdiff(pred_lower, cond_lower)  
  missing_lower <- setdiff(cond_lower, pred_lower)  
  shd <- length(extra_lower) + length(missing_lower) + length(reverse)  
  
  precision <- 1 - fdr  
  recall <- tpr  
  f1 <- 2 * precision * recall / max((precision + recall), 1)  
  
  # Output and logging  
  cat(sprintf("fdr: %.3f, tpr: %.3f, fpr: %.3f, shd: %d, nnz: %d, f1: %.3f\n",  
              fdr, tpr, fpr, shd, pred_size, f1))  
  
  return(c(fdr = fdr, tpr = tpr, fpr = fpr, shd = shd, nnz = pred_size, f1 = f1))  
}  

count_accuracy_und <- function(B_true, B_est) {  
  # Validate function arguments  
  if (!all(B_est %in% c(0, 1))) {  
    stop("B_est should take value in {0,1}")  
  }  
  
  d <- nrow(B_true)  
  # Linear index of non-zero elements  
  pred <- which(B_est != 0)  
  cond <- which(B_true != 0)  
  cond_reversed <- which(t(B_true) != 0)  
  cond_skeleton <- unique(c(cond, cond_reversed))  
  
  # True positives, false positives, and reversals  
  true_pos <- intersect(pred, cond)  
  false_pos <- setdiff(pred, cond_skeleton)  
  extra <- setdiff(pred, cond)  
  reverse <- intersect(extra, cond_reversed)  
  
  # Compute accuracy metrics  
  pred_size <- length(pred)  
  cond_neg_size <- 0.5 * d * (d - 1) - length(cond)  
  fdr_und <- (length(false_pos)) / max(pred_size, 1)  
  tpr_und <- (length(reverse) + length(true_pos)) / max(length(cond), 1)  
  fpr_und <- (length(false_pos)) / max(cond_neg_size, 1)  
  
  # Get lower triangle matrix  
  get_lower_triangle <- function(mat) {  
    if (!is.matrix(mat)) {  
      stop("Input must be a matrix.")  
    }  
    
    # Create a matrix with the same dimensions as the input matrix, initialized to zero  
    lower_triangle <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))  
    
    # Copy the lower triangle values to the new matrix  
    lower_triangle[lower.tri(mat, diag = TRUE)] <- mat[lower.tri(mat, diag = TRUE)]  
    
    return(lower_triangle)  
  }  
  
  pred_lower <- which(get_lower_triangle(B_est + t(B_est)) != 0)  
  cond_lower <- which(get_lower_triangle(B_true + t(B_true)) != 0)  
  extra_lower <- setdiff(pred_lower, cond_lower)  
  missing_lower <- setdiff(cond_lower, pred_lower)  
  shd_und <- length(extra_lower) + length(missing_lower)  
  
  precision_und <- 1 - fdr_und  
  recall_und <- tpr_und  
  f1_und <- 2 * precision_und * recall_und / max((precision_und + recall_und), 1)  
  
  # Output and logging  
  cat(sprintf("fdr_und: %.3f, tpr_und: %.3f, fpr_und: %.3f, shd_und: %d, nnz: %d, f1_und: %.3f\n",  
              fdr_und, tpr_und, fpr_und, shd_und, pred_size, f1_und))  
  
  return(c(fdr_und = fdr_und, tpr_und = tpr_und, fpr_und = fpr_und,   
           shd_und = shd_und, nnz = pred_size, f1_und = f1_und))  
}  

# HC (treat all loss_type as continuous) ------------------------  
library(bnlearn)  

# Define HCtoW function to learn the Bayesian network structure and return adjacency matrix estimate W  
HCtoW <- function(x) {  
  hc_net = bnlearn::hc(x, whitelist = NULL, blacklist = NULL,debug = FALSE)
  nodes <- paste0("V", 1:ncol(x))  
  
  W <- as.matrix(matrix(0, ncol = length(nodes),   
                        nrow = length(nodes)))  
  for (i in 1:nrow(hc_net$arcs)) {  
    from <- which(nodes == hc_net$arcs[i, 1])  
    to <- which(nodes == hc_net$arcs[i, 2])  
    W[from, to] <- 1  
  }  
  return(W)  
}  

# Define path and filename  
default_save_path <- "./HC"  

# Create empty data frames to store results  
results <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  i = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)   
results_avg <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)   

# Loop for different k 
for (n in c(1000)) {  
  for (k in c(2,4,10)) {   
    print(paste0("k=",k))  
    library(tidyverse)  
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_k%d.csv", n, k))
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    
    for (i in 0:9) {  
      print(paste0("i=", i))  
      
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_k%d_i%d.csv", n, k, i))  
      filenameX <- file.path(default_save_path, sprintf("X_n%d_k%d_i%d.csv", n, k, i))        
      X <- read.csv(filenameX, header = FALSE)   
      W_est <- HCtoW(X)  
      
      write.csv(W_est, filenameWE)  
      ACC[i + 1, ] <- count_accuracy(B_true, W_est)  
      ACC_und[i + 1, ] <- count_accuracy_und(B_true, W_est)  
      
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         k = k,  
                         i = i,  
                         fdr = ACC[i + 1, ][1],   
                         tpr = ACC[i + 1, ][2],   
                         fpr = ACC[i + 1, ][3],   
                         shd = ACC[i + 1, ][4],   
                         pred_size = ACC[i + 1, ][5],   
                         F1_score = ACC[i + 1, ][6],   
                         fdr_und = ACC_und[i + 1, ][1],   
                         tpr_und = ACC_und[i + 1, ][2],   
                         fpr_und = ACC_und[i + 1, ][3],   
                         shd_und = ACC_und[i + 1, ][4],   
                         pred_size = ACC_und[i + 1, ][5],   
                         F1_score_und = ACC_und[i + 1, ][6]   
                       )   
      )   
      
      cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    acc <- colMeans(ACC)  
    acc_und <- colMeans(ACC_und)  
    cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(acc, collapse = ",")))  
    cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("k=%d execution time: %.2f seconds\n", k, exe_time))
    
    results_avg <- rbind(results_avg,  
                         data.frame(  
                           n = n,  
                           k = k,  
                           fdr = acc[1],   
                           tpr = acc[2],   
                           fpr = acc[3],   
                           shd = acc[4],   
                           pred_size = acc[5],   
                           F1_score = acc[6],   
                           fdr_und = acc_und[1],   
                           tpr_und = acc_und[2],   
                           fpr_und = acc_und[3],   
                           shd_und = acc_und[4],   
                           pred_size = acc_und[5],   
                           F1_score_und = acc_und[6]   
                         )   
    )  
  }  
}  

# Write results to CSV files  
write.csv(results_avg, file.path(default_save_path, "acc_k_HC.csv"), row.names = FALSE)   
write.csv(results, file.path(default_save_path, "each_acc_k_HC.csv"), row.names = FALSE)   


# TABU (treat all loss_type as continuous) ------------------------  
library(bnlearn)  

# Define TABUtoW function to learn the Bayesian network structure and return adjacency matrix estimate W  
TABUtoW <- function(x) {  
  tabu_net = bnlearn::tabu(x, whitelist = NULL, blacklist = NULL,debug = FALSE)
  nodes <- paste0("V", 1:ncol(x))  
  
  W <- as.matrix(matrix(0, ncol = length(nodes),   
                        nrow = length(nodes)))  
  for (i in 1:nrow(tabu_net$arcs)) {  
    from <- which(nodes == tabu_net$arcs[i, 1])  
    to <- which(nodes == tabu_net$arcs[i, 2])  
    W[from, to] <- 1  
  }  
  return(W)  
}  

# Define path and filename  
default_save_path <- "./TABU"  

# Create empty data frames to store results  
results <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  i = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)   
results_avg <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)   

# Loop for different k 
for (n in c(1000)) {  
  for (k in c(2,4,10)) {   
    print(paste0("k=",k))  
    library(tidyverse)  
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_k%d.csv", n, k))
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    
    for (i in 0:9) {  
      print(paste0("i=", i))  
      
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_k%d_i%d.csv", n, k, i))  
      filenameX <- file.path(default_save_path, sprintf("X_n%d_k%d_i%d.csv", n, k, i))        
      X <- read.csv(filenameX, header = FALSE)   
      W_est <- TABUtoW(X)  
      
      write.csv(W_est, filenameWE)  
      ACC[i + 1, ] <- count_accuracy(B_true, W_est)  
      ACC_und[i + 1, ] <- count_accuracy_und(B_true, W_est)  
      
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         k = k,  
                         i = i,  
                         fdr = ACC[i + 1, ][1],   
                         tpr = ACC[i + 1, ][2],   
                         fpr = ACC[i + 1, ][3],   
                         shd = ACC[i + 1, ][4],   
                         pred_size = ACC[i + 1, ][5],   
                         F1_score = ACC[i + 1, ][6],   
                         fdr_und = ACC_und[i + 1, ][1],   
                         tpr_und = ACC_und[i + 1, ][2],   
                         fpr_und = ACC_und[i + 1, ][3],   
                         shd_und = ACC_und[i + 1, ][4],   
                         pred_size = ACC_und[i + 1, ][5],   
                         F1_score_und = ACC_und[i + 1, ][6]   
                       )   
      )   
      
      cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    acc <- colMeans(ACC)  
    acc_und <- colMeans(ACC_und)  
    cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(acc, collapse = ",")))  
    cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("k=%d execution time: %.2f seconds\n", k, exe_time))
    
    results_avg <- rbind(results_avg,  
                         data.frame(  
                           n = n,  
                           k = k,  
                           fdr = acc[1],   
                           tpr = acc[2],   
                           fpr = acc[3],   
                           shd = acc[4],   
                           pred_size = acc[5],   
                           F1_score = acc[6],   
                           fdr_und = acc_und[1],   
                           tpr_und = acc_und[2],   
                           fpr_und = acc_und[3],   
                           shd_und = acc_und[4],   
                           pred_size = acc_und[5],   
                           F1_score_und = acc_und[6]   
                         )   
    )  
  }  
}  

# Write results to CSV files  
write.csv(results_avg, file.path(default_save_path, "acc_k_TABU.csv"), row.names = FALSE)   
write.csv(results, file.path(default_save_path, "each_acc_k_TABU.csv"), row.names = FALSE)   

# mDAG -------------------------------------------------------------  
library(mDAG)  

# Define loss type ('g' for 'gauss' and 'c' for 'categorical')
get_loss_type <- function(k) {  
  if (k == 2) {  
    return(c(rep("g", 12), "c", rep("g", 5), "c", "g"))  
  } else if (k == 4) {  
    return(c(rep("g", 12), "c", "g", "c", rep("g", 3), rep("c", 2)))  
  } else if (k == 10) {  
    return(c("g", "c", "g", "c", "c", "g", "g", "g", "c", "g", 
             "g", "g", "c", "g", "c", "g", "c", "c", "c", "c"))  
  } else {  
    stop(paste("Invalid value for k:", k))  
  }  
} 
# Define levels for categorical nodes
get_level <- function(k) {  
  loss_type_str <- get_loss_type(k)  
  loss_type_num <- rep(0, length(loss_type_str))  
  loss_type_num[loss_type_str == "g"] <- 1  
  loss_type_num[loss_type_str == "c"] <- 2  
  return(loss_type_num)  
}

# Define mDAGtoW function to learn the Bayesian network structure and return adjacency matrix estimate W
mDAGtoW <- function(data, type, level) {  
  d <- dim(data)[2]  
  dag = mDAG(data = data, type = type, level = level, nperm = 150)  
  print(dag$skeleton)  
  return(dag$skeleton)  
}  

# Define path and filename 
default_save_path <- "./mDAG"  

# Create empty data frames to store results  
results <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  i = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)  

results_avg <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)  

# Loop for different k
for (n in c(1000)) {  
  for (k in c(2, 4, 10)) {   
    print(paste0("k=",k))
    library(tidyverse)  
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_k%d.csv", n, k))
    
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    for (i in 0:9) {  
      print(i)
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_k%d_i%d.csv", n, k, i))  
      filenameX <- file.path(default_save_path, sprintf("X_n%d_k%d_i%d.csv", n, k, i))  
      
      
      X <- read.csv(filenameX, header = FALSE)  
      W_est <- mDAGtoW(data = X,   
                       type = get_loss_type(k),   
                       level = get_level(k))   
      
      write.csv(W_est, filenameWE)  
      ACC[i + 1, ] <- count_accuracy(B_true, W_est)  
      ACC_und[i + 1, ] <- count_accuracy_und(B_true, W_est)  
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         k = k,  
                         i = i,  
                         fdr = ACC[i + 1, ][1],   
                         tpr = ACC[i + 1, ][2],   
                         fpr = ACC[i + 1, ][3],   
                         shd = ACC[i + 1, ][4],   
                         pred_size = ACC[i + 1, ][5],   
                         F1_score = ACC[i + 1, ][6],   
                         fdr_und = ACC_und[i + 1, ][1],   
                         tpr_und = ACC_und[i + 1, ][2],   
                         fpr_und = ACC_und[i + 1, ][3],   
                         shd_und = ACC_und[i + 1, ][4],   
                         pred_size = ACC_und[i + 1, ][5],   
                         F1_score_und = ACC_und[i + 1, ][6]  
                       )   
      )   
      
      cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    acc <- colMeans(ACC)  
    acc_und <- colMeans(ACC_und)  
    cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(acc, collapse = ",")))  
    cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("k=%d execution time: %.2f seconds\n", k, exe_time))
    
    results_avg <- rbind(results_avg,  
                         data.frame(  
                           n = n,  
                           k = k,  
                           fdr = acc[1],   
                           tpr = acc[2],   
                           fpr = acc[3],   
                           shd = acc[4],   
                           pred_size = acc[5],   
                           F1_score = acc[6],   
                           fdr_und = acc_und[1],   
                           tpr_und = acc_und[2],   
                           fpr_und = acc_und[3],   
                           shd_und = acc_und[4],   
                           pred_size = acc_und[5],   
                           F1_score_und = acc_und[6]  
                         )   
    )  
  }  
}  

# Write results to CSV files  
write.csv(results_avg, file.path(default_save_path, "acc_k_mDAG.csv"), row.names = FALSE)   
write.csv(results, file.path(default_save_path, "each_acc_k_mDAG.csv"), row.names = FALSE)   

# DAGBagM (baseline version) -------------------------------------------------------  
# detach("package:bnlearn", unload = TRUE)  
# detach("package:mDAG", unload = TRUE)  
library(dagbagM)  

# Define loss type ('c' for 'continuous' and 'b' for 'binary')
get_loss_type <- function(k) {
  if (k == 2) {
    return(c(rep("c", 12), "b", rep("c", 5), "b", "c"))
  } else if (k == 4) {
    return(c(rep("c", 12), "b", "c", "b", rep("c", 3), rep("b", 2)))
  } else if (k == 10) {
    return(c("c", "b", "c", "b", "b", "c", "c", "c", "b", "c", "c", "c", "b", "c", "b", "c", "b", "b", "b", "b"))
  } else {
    stop(paste("Invalid value for k:", k))
  }
}
# Define path and filename
default_save_path <- "./DAGBagM"  

# Define mDAGtoW function to learn the Bayesian network structure and return adjacency matrix estimate W
dagbagMtoW <- function(X, nodeType) {  
  hc_res <- hc(Y = X, nodeType = nodeType)  
  W_est <- hc_res$adjacency + 0  
  return(W_est)  
}  

# Create empty data frames to store results  
results <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  i = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)  

results_avg <- data.frame(  
  n = numeric(),  
  k = numeric(),  
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),  
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()  
)  

# Loop for different k
for (n in c(1000)) {  
  for (k in c(2, 4, 10)) {
    print(paste0("k=",k))
    library(tidyverse)  
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_k%d.csv", n, k))
    
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    
    for (i in 0:9) {  
      print(i)  
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_k%d_i%d.csv", n, k, i))  
      filenameX <- file.path(default_save_path, sprintf("X_n%d_k%d_i%d.csv", n, k, i))
      X <- read.csv(filenameX, header = FALSE)  %>% as.matrix()
      W_est <- dagbagMtoW(X, nodeType = get_loss_type(k))  
      
      write.csv(W_est, filenameWE)  
      ACC[i + 1, ] <- count_accuracy(B_true, W_est)  
      ACC_und[i + 1, ] <- count_accuracy_und(B_true, W_est)  
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         k = k,  
                         i = i,  
                         fdr = ACC[i + 1, ][1],   
                         tpr = ACC[i + 1, ][2],   
                         fpr = ACC[i + 1, ][3],   
                         shd = ACC[i + 1, ][4],   
                         pred_size = ACC[i + 1, ][5],   
                         F1_score = ACC[i + 1, ][6],   
                         fdr_und = ACC_und[i + 1, ][1],   
                         tpr_und = ACC_und[i + 1, ][2],   
                         fpr_und = ACC_und[i + 1, ][3],   
                         shd_und = ACC_und[i + 1, ][4],   
                         pred_size = ACC_und[i + 1, ][5],   
                         F1_score_und = ACC_und[i + 1, ][6]  
                       )   
      )   
      
      cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    acc <- colMeans(ACC)  
    acc_und <- colMeans(ACC_und)  
    cat(sprintf("n=%d,k=%d,acc=%s\n", n, k, paste(acc, collapse = ",")))  
    cat(sprintf("n=%d,k=%d,acc_und=%s\n", n, k, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("k=%d execution time: %.2f seconds\n", k, exe_time))    
    
    results_avg <- rbind(results_avg,  
                         data.frame(  
                           n = n,  
                           k = k,  
                           fdr = acc[1],   
                           tpr = acc[2],   
                           fpr = acc[3],   
                           shd = acc[4],   
                           pred_size = acc[5],   
                           F1_score = acc[6],   
                           fdr_und = acc_und[1],   
                           tpr_und = acc_und[2],   
                           fpr_und = acc_und[3],   
                           shd_und = acc_und[4],   
                           pred_size = acc_und[5],   
                           F1_score_und = acc_und[6]  
                         )   
    )  
  }  
}  

# Write results to CSV files  
write.csv(results_avg, file.path(default_save_path, "acc_k_DAGBagM.csv"), row.names = FALSE)   
write.csv(results, file.path(default_save_path, "each_acc_k_DAGBagM.csv"), row.names = FALSE)