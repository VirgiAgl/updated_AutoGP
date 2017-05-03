library(languageR)
library(ggplot2)
library(plotrix)
library(plyr)
library(party)
library(gridExtra)
library(Hmisc)
library(extrafont)
library(scales)
library(reshape)
library(pbkrtest)
library(nloptr)
library(optimx)
library(data.table)
library(extrafont)
library(dgof)
loadfonts()

#some helper functions
source('Workspace/updated_AutoGP/R_plots/helper.R')
source('Workspace/updated_AutoGP/R_plots/plotting_functions.R')

SP_name = "SF"

w = 15
h = 5
sp = 1

output_path = "Workspace/updated_AutoGP/R_plots"

# synthetic point process data
name = 'synthetic_data'
inputs = read.csv('Workspace/updated_AutoGP/R_plots/data_inputs.csv')
outputs = read.csv('Workspace/updated_AutoGP/R_plots/data_outputs.csv')
posterior_mean_intensity = read.csv('Workspace/updated_AutoGP/R_plots/total_results_ypred.csv')
posterior_var_intensity = read.csv('Workspace/updated_AutoGP/R_plots/total_results_postvar.csv')
xtest = read.csv('Workspace/updated_AutoGP/R_plots/xtest.csv')
ytest = read.csv('Workspace/updated_AutoGP/R_plots/ytest.csv')
ytrain = read.csv('Workspace/updated_AutoGP/R_plots/ytrain.csv')
xtrain = read.csv('Workspace/updated_AutoGP/R_plots/xtrain.csv')
sample_intensity_test= read.csv('Workspace/updated_AutoGP/R_plots/sample_intensity_test.csv')

# N = 200
# idx = as.integer(unlist(idx))
# input_indices = sort(idx[(N):length(idx)])
# inputs$id = c(1:nrow(inputs))
# xtest = inputs[inputs$id %in% input_indices,]

#data$model = toupper(substr(data$model_sp,0, 4))
#data = rename_model(data)
#data$sp = paste(SP_name, "=", substr(data$model_sp,6, 8))

data_intensity = cbind(xtest, posterior_mean_intensity, posterior_var_intensity, sample_intensity_test)
colnames(data_intensity) = c("x","m","v", "sample_intensity")
data = cbind(inputs,outputs)
colnames(data) = c("x", "y")

model = "FG"
y_lab = "intensity"

p1 = draw_mining_data(data)

p2 = ggplot(data_intensity, aes(x=x, y = m, colour = 1)) + 
  geom_point() +
  geom_ribbon(aes(x=x, ymin= m - 2 * sqrt(v), ymax=m + 2 * sqrt(v)), fill="grey", alpha=.4, colour =NA) +  
  xlab('') +
  ylab(y_lab) +
  theme_bw() +

  theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical", 
        axis.line = element_line(colour = "black"),
        panel.grid.minor=element_blank(),      
        panel.border = element_blank(),
        panel.spacing = unit(.4, "lines"),
        text=element_text(family="Arial", size=10),
        legend.key = element_blank(),
        strip.background = element_rect(colour = "white", fill = "white",
                                        size = 0.5, linetype = "solid"),
        axis.ticks.x = element_blank(),
        legend.title=element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1)
) 
  
compare_intensity = HistPlot(list(data_intensity$m, data_intensity$sample_intensity), method = c("Posterior mean intensity","Initial intensity"), ggtitle = "Distributions for the intensity of the process")
#ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      

# Kolmogorov Smirnov test. The smaller the p-value, the less likely that d1=d0
ks.test(data_intensity$m, data_intensity$sample_intensity)
# Wilcoxon signed-rank test
wilcox.test(data_intensity$m, data_intensity$sample_intensity)
