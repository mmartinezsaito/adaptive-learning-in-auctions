if (!require(RColorBrewer)) install.packages("RColorBrewer"); library(RColorBrewer)
if (!require(dplyr)) install.packages("dplyr"); library(dplyr)
library(lattice)
library(ggplot2)
library(knitr)
library(multcomp)


## Read data of behavioral control in Volgogradsky Prospekt 46B, 323
cwd2 <- "~/Dropbox/Econoshyuka/logs/"
setwd(cwd2)
h <- list()
for (i in list.files())
  if (grepl("^market_VPN_[bcd].*", i)) {
    h <- rbind(h, read.table(i, skip=1))  
  }
names(h) <- c("sid", "block_n", "snb", "bid", "b2_bid", "s1_rp", "s2_rp", "outcome", "rt", "profit")
h$b2_bid <- as.numeric(levels(h$b2_bid))[h$b2_bid]
h$s2_rp <- as.numeric(levels(h$s2_rp))[h$s2_rp]
h$rt <- as.numeric(h$rt)
attr(h$snb, "levels") <- c("BC","SC","NC"); h$snb <- factor(h$snb,levels(h$snb)[c(2,3,1)]) 
h$outcome[h$outcome=="none"] <- NA
h <- mutate(h, out_bool = as.logical(ifelse(outcome=="accept", 1, 0)))
for (s in levels(h$sid)) {
  isbey24 <- F
  for (i in which(h$sid==s)) {
    if (isbey24) h$block_n[i] <- h$block_n[i] + 24
    else if (h$block_n[i] == 24 && h$block_n[i+1] == 1) isbey24 <- T        
  }
}
#h$cond <- factor(ifelse(grepl("^b.*", h$sid), "b", "c"))
h$cond <- factor(regmatches(h$sid, regexpr("^([b-d]).*?", h$sid)))

qd2.colc <- c("factor", "factor", "Date", "integer", "factor", "factor", "factor", 
              "factor", "factor", "factor", "factor", "factor", "Date", "factor")
#qd2 <- read.table("questionnaire2.txt", header = T, sep = "\t", skip = 0, colClasses=qd2.colc)
#qd2 <- read.table("q2", header = T, sep = "\t", skip = 0, colClasses=qd2.colc)
#qd2 <- subset(qd2, select=-sn)
#qd2$wealth <- ordered(qd2$wealth, levels=c("very_low", "low", "medium", "high"))


## r defining some utils
bwp_out <- function(d, r)  {   
  pb <- d[r,]$block_n - 1; m <- d[r,]$snb; s <- d[r,]$sid;    
  po <- d[d$block_n==pb & d$snb==m & d$sid==s,]$out_bool;   
  if (length(po)!=0) return(po)   
  else return(NA) 
}
bwap_out <- function(d, r)  {   
  if (r<2) return(NA)
  rn <- as.numeric(row.names(d[r,])); s <- d[r,]$sid;    
  po <- d[rn - 1, ]$out_bool;   
  if (length(po)!=0 & d[rn - 1,]$sid == s) return(po)   
  else return(NA) 
}
bwp_bid <- function(d, r) {
  pb <- d[r,]$block_n - 1; m <- d[r,]$snb; s <- d[r,]$sid;    
  pbi <- d[d$block_n==pb & d$snb==m & d$sid==s,]$bid;   
  if (length(pbi)!=0) return(pbi)   
  else return(NA)
} 
bwap_bid <- function(d, r) {   
  if (r<2) return(NA)
  rn <- as.numeric(row.names(d[r,])); s <- d[r,]$sid;    
  pbi <- d[rn - 1, ]$bid;     
  if (r>1 & length(pbi)!=0 & d[rn-1,]$sid == s) return(pbi)   
  else return(NA) 
}
bwn_bid <- function(d, r) {
  nb <- d[r,]$block_n + 1; m <- d[r,]$snb; s <- d[r,]$sid;    
  nbi <- d[d$block_n==nb & d$snb==m & d$sid==s,]$bid;   
  if (length(nbi)!=0) return(nbi)   
  else return(NA)
}
bwan_bid <- function(d, r) {   
  if (r>nrow(h)-1) return(NA)
  rn <- as.numeric(row.names(d[r,])); s <- d[r,]$sid;    
  nbi <- d[rn + 1, ]$bid;     
  if (length(nbi)!=0 & d[rn+1,]$sid == s) return(nbi)   
  else return(NA) 
} 
dl_splitter <- function(d, r) {  # split trails by degree of compliance to DL
  pb <- d[r,]$block_n - 1; m <- d[r,]$snb; s <- d[r,]$sid;    
  cbi <- d[r,]$bid;
  pbi <- d[d$block_n==pb & d$snb==m & d$sid==s,]$bid;   
  pout_bool <- d[d$block_n==pb & d$snb==m & d$sid==s,]$out_bool; 
  if (length(pbi)!=0) return((cbi>pbi & !pout_bool) | (cbi<=pbi & pout_bool))   
  else return(NA)
}

## Create features conditioned on previous trials also for control study
h$out_prev <- rep(0, nrow(h)) 
for (i in 1:nrow(h)) h$out_prev[[i]] <- as.logical(bwp_out(h, i)) 
h$out_aprev <- rep(0, nrow(h)) 
for (i in 1:nrow(h)) h$out_aprev[[i]] <- as.logical(bwap_out(h, i)) 
h$bid_prev <- rep(0, nrow(h)) 
for (i in 1:nrow(h)) h$bid_prev[[i]] <- as.numeric(bwp_bid(h, i))  
h$bid_aprev <- rep(0, nrow(h)) 
for (i in 1:nrow(h)) h$bid_aprev[[i]] <- as.numeric(bwap_bid(h, i))
h$bid_next <- rep(0, nrow(h)) 
for (i in 1:nrow(h)) h$bid_next[[i]] <- as.numeric(bwn_bid(h, i))
h$bid_anext <- rep(0, nrow(h)) 
for (i in 1:nrow(h)) h$bid_anext[[i]] <- as.numeric(bwan_bid(h, i))
h$dl <- rep(0, nrow(h)) 
for (i in 1:nrow(h)) h$dl[[i]] <- as.numeric(dl_splitter(h, i))
#if ("package:dplyr" %in% search()) write.csv(h, "econoshyuka.csv")


## Plots

library(KernSmooth)
#library(ks)

### Control experiments
h2 <- h
h24 <- h[h$block_n < 25,]
h25 <- h[h$block_n > 24,]
nl <- c("b","c","d")
condls <- c("T1","T2","T3")
names(condls) <- nl
ml <- c("SC", "NC", "BC")
cl0 <- c(rgb(1,0,0,0.2),rgb(0,1,0,0.2),rgb(0,0,1,0.2))
cl1 <- c(rgb(1,0,0,1),rgb(0,1,0,1),rgb(0,0,1,1))

bwplot(profit ~ snb | cond, ylab="Profit", data=h)
bwplot(bid ~ snb | cond, ylab="bid", data=h)


## First trials bids
h2b1 <- h[h$block_n==1,]
for (i in nl) {
  print(summary(aov(bid ~ snb, data=h2b1[h2b1$cond==i,])))
  print(tapply(h2b1[h2b1$cond==i,]$bid, h2b1[h2b1$cond==i,]$snb, function(x) mean(x, na.rm=T)))
  print(sapply(split(h2b1[h2b1$cond==i,]$bid, h2b1[h2b1$cond==i,]$snb), function(x) sd(x, na.rm=T)))
  cat('\n')
}
# fig03
tiff("fig03.tif", width=88, height=70, units="mm", pointsize=8, res=300)
g <- ggplot(data = h2b1, mapping = aes(x=snb, y=bid))
g <- g + stat_boxplot(geom="errorbar") + stat_boxplot(geom="boxplot", size=1) #geom_boxplot()
g <- g + facet_grid(.~cond, labeller = as_labeller(c('b'="T1",'c'="T2",'d'="T3"))) 
g <- g + theme_bw() + theme(strip.background=element_blank()) #rect(fill="white"))
g <- g + xlab("Market type") + ylab("Bid") + coord_cartesian(ylim = c(0, 10))
print(g)
dev.off()


## Some stats
hp<-h24
table(hp$out_bool, hp$snb, hp$cond)
table(hp$out_bool, hp$cond)

table(hp$out_prev, hp$dl, hp$cond)
table(hp$dl, hp$cond)

table(cut(h2$profit,breaks=seq(1,10)), h2$snb, h2$cond)
tapply(h2$profit, h2$snb, function(x) sd(x, na.rm=T)) #sapply(split(h$profit, h$snb), function(x) sd(x, na.rm=T))
sapply(split(h2$bid, h2$snb), function(x) {x<-x[!is.na(x)];n<-length(x)-1; sqrt(sd(x)^2*n/qchisq(c(0.05,0.5,0.95), df=n, lower.tail=F))})


##   bid - blocknum trends
sc_mfx_a <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='SC',], na.action = na.omit)
nc_mfx_a <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='NC',], na.action = na.omit)
bc_mfx_a <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='BC',], na.action = na.omit)
sc_mfx_b <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='SC' & h$cond=='b',], na.action = na.omit)
nc_mfx_b <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='NC' & h$cond=='b',], na.action = na.omit)
bc_mfx_b <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='BC' & h$cond=='b',], na.action = na.omit)
sc_mfx_c <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='SC' & h$cond=='c',], na.action = na.omit)
nc_mfx_c <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='NC' & h$cond=='c',], na.action = na.omit)
bc_mfx_c <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='BC' & h$cond=='c',], na.action = na.omit)
sc_mfx_d <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='SC' & h$cond=='d',], na.action = na.omit)
nc_mfx_d <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='NC' & h$cond=='d',], na.action = na.omit)
bc_mfx_d <- lme(fixed = bid ~ block_n, random = ~ 1|sid, data = h[h$snb=='BC' & h$cond=='d',], na.action = na.omit)

# 01-24: fig04 
tiff("fig04.tif", width=88, height=44, units="mm", pointsize=8, res=300)
par(mfrow=c(1,3), mar=c(4,4,2,.1), pch=16)
sc_ffx_b <- lm(bid ~ block_n, data = hp[hp$snb=='SC' & hp$cond=='b',], na.action = na.omit)
nc_ffx_b <- lm(bid ~ block_n, data = hp[hp$snb=='NC' & hp$cond=='b',], na.action = na.omit)
bc_ffx_b <- lm(bid ~ block_n, data = hp[hp$snb=='BC' & hp$cond=='b',], na.action = na.omit)
plot(bid ~ block_n, col=cl0[hp$snb], data=hp[hp$cond=='b',], xlab="Trial", ylab="Bid", main="T1")
abline(sc_ffx_b, lwd=2, col=cl0[1]); abline(nc_ffx_b, lwd=2, col=cl1[2]); abline(bc_ffx_b, lwd=2, col=cl1[3])
legend("bottomleft", legend=ml, pch=16, col=cl1, text.col="black", bty="n")
sc_ffx_c <- lm(bid ~ block_n, data = hp[hp$snb=='SC' & hp$cond=='c',], na.action = na.omit)
nc_ffx_c <- lm(bid ~ block_n, data = hp[hp$snb=='NC' & hp$cond=='c',], na.action = na.omit)
bc_ffx_c <- lm(bid ~ block_n, data = hp[hp$snb=='BC' & hp$cond=='c',], na.action = na.omit)
plot(bid ~ block_n, col=cl0[hp$snb], data=hp[hp$cond=='c',], xlab="Trial", ylab="", ylim=c(0,10), main="T2")
abline(sc_ffx_c, lwd=2, col=cl1[1]); abline(nc_ffx_c, lwd=2, col=cl1[2]); abline(bc_ffx_c, lwd=2, col=cl1[3])
sc_ffx_d <- lm(bid ~ block_n, data = hp[hp$snb=='SC' & hp$cond=='d',], na.action = na.omit)
nc_ffx_d <- lm(bid ~ block_n, data = hp[hp$snb=='NC' & hp$cond=='d',], na.action = na.omit)
bc_ffx_d <- lm(bid ~ block_n, data = hp[hp$snb=='BC' & hp$cond=='d',], na.action = na.omit)
plot(bid ~ block_n, col=cl0[hp$snb], data=hp[hp$cond=='d',], xlab="Trial", ylab="", main="T3")
abline(sc_ffx_d, lwd=2, col=cl1[1]); abline(nc_ffx_d, lwd=2, col=cl1[2]); abline(bc_ffx_d, lwd=2, col=cl1[3])
dev.off()

# 25-40
sc_ffx_b <- lm(bid ~ block_n, data = h25[h25$snb=='SC' & h25$cond=='b',], na.action = na.omit)
nc_ffx_b <- lm(bid ~ block_n, data = h25[h25$snb=='NC' & h25$cond=='b',], na.action = na.omit)
bc_ffx_b <- lm(bid ~ block_n, data = h25[h25$snb=='BC' & h25$cond=='b',], na.action = na.omit)
plot(bid ~ block_n, col=h25$snb, data=h25[h25$cond=='b',], xlab="block number")
abline(sc_ffx_b, lwd=3); abline(nc_ffx_b, lwd=3); abline(bc_ffx_b, lwd=3)
sc_ffx_c <- lm(bid ~ block_n, data = h25[h25$snb=='SC' & h25$cond=='c',], na.action = na.omit)
nc_ffx_c <- lm(bid ~ block_n, data = h25[h25$snb=='NC' & h25$cond=='c',], na.action = na.omit)
bc_ffx_c <- lm(bid ~ block_n, data = h25[h25$snb=='BC' & h25$cond=='c',], na.action = na.omit)
plot(bid ~ block_n, col=h25$snb, data=h25[h25$cond=='c',], xlab="block number", ylab="")
abline(sc_ffx_c, lwd=3); abline(nc_ffx_c, lwd=3); abline(bc_ffx_c, lwd=3)
sc_ffx_d <- lm(bid ~ block_n, data = h25[h25$snb=='SC' & h25$cond=='d',], na.action = na.omit)
nc_ffx_d <- lm(bid ~ block_n, data = h25[h25$snb=='NC' & h25$cond=='d',], na.action = na.omit)
bc_ffx_d <- lm(bid ~ block_n, data = h25[h25$snb=='BC' & h25$cond=='d',], na.action = na.omit)
plot(bid ~ block_n, col=h$snb, data=h[h$cond=='d',], xlab="block number", ylab="")
abline(sc_ffx_d, lwd=3); abline(nc_ffx_d, lwd=3); abline(bc_ffx_d, lwd=3)


## 40x3 boxplots for condition b and c
par(mfrow = c(1,3))
boxplot(split(h$bid[h$snb=="SC"&h$cond=="b"], h$block_n[h$snb=="SC"&h$cond=="b"]), col=rgb(1,0,0,0.7), ylim= c(1,9.5), notch=T, varwidth=T)
boxplot(split(h$bid[h$snb=="NC"&h$cond=="b"], h$block_n[h$snb=="NC"&h$cond=="b"]), col=rgb(0,1,0,0.7), notch=T, varwidth=T, add=T)
boxplot(split(h$bid[h$snb=="BC"&h$cond=="b"], h$block_n[h$snb=="BC"&h$cond=="b"]), col=rgb(0,0,1,0.7), notch=T, varwidth=T, add=T)
boxplot(split(h$bid[h$snb=="SC"&h$cond=="c"], h$block_n[h$snb=="SC"&h$cond=="c"]), col=rgb(0,1,0,0.2), notch=T, varwidth=T, add=T, border="white")
boxplot(split(h$bid[h$snb=="NC"&h$cond=="c"], h$block_n[h$snb=="NC"&h$cond=="c"]), col=rgb(0,1,0,0.2), notch=T, varwidth=T, add=T, border="white")
boxplot(split(h$bid[h$snb=="BC"&h$cond=="c"], h$block_n[h$snb=="BC"&h$cond=="c"]), col=rgb(0,0,1,0.2), notch=T, varwidth=T, add=T, border="white")

par(mfrow = c(1,3))
boxplot(split(h$bid[h$snb=="SC"&h$cond=="b"], h$block_n[h$snb=="SC"&h$cond=="b"]), col=rgb(1,0,0,0.7), ylim= c(1,9.5), notch=T, varwidth=T)
boxplot(split(h$bid[h$snb=="SC"&h$cond=="c"], h$block_n[h$snb=="SC"&h$cond=="c"]), col=rgb(0,1,0,0.2), notch=T, varwidth=T, add=T)
boxplot(split(h$bid[h$snb=="NC"&h$cond=="b"], h$block_n[h$snb=="NC"&h$cond=="b"]), col=rgb(0,1,0,0.7), ylim= c(1,9.5), notch=T, varwidth=T)
boxplot(split(h$bid[h$snb=="NC"&h$cond=="c"], h$block_n[h$snb=="NC"&h$cond=="c"]), col=rgb(0,1,0,0.2), notch=T, varwidth=T, add=T)
boxplot(split(h$bid[h$snb=="BC"&h$cond=="b"], h$block_n[h$snb=="BC"&h$cond=="b"]), col=rgb(0,1,0,0.7), ylim= c(1,9.5), notch=T, varwidth=T)
boxplot(split(h$bid[h$snb=="BC"&h$cond=="c"], h$block_n[h$snb=="BC"&h$cond=="c"]), col=rgb(0,1,0,0.2), notch=T, varwidth=T, add=T)

bwplot(bid ~ block_n | cond + snb, xlab=c("Block number"), ylab="Bid", data=h, horizontal=FALSE)
coplot(bid ~ block_n | snb, data=h, col=as.numeric(h$cond), columns=3)

# fig05
hpg<-hp; hpg$bid[hp$cond=="c"]<-hp$bid[hp$cond=="b"]; hpg$bid[hp$cond=="d"]<-hp$bid[hp$cond=="b"]
tiff("fig05.tif", width=99, height=70, units="mm", pointsize=8, res=300)
g <- ggplot(data = hp, mapping = aes(x=block_n, y=bid, color=snb))
g <- g + geom_point(alpha=0.2, size=.5) + geom_smooth(method="loess", se=TRUE, level=0.95)
#g <- g + geom_smooth(color="#B0B0B0", method="loess", se=TRUE, level=0.95)
#g <- g + geom_smooth(data=hpg, aes(y=bid, color="#808080"), method="loess", se=TRUE, level=0.95)
#g <- g + geom_segment(x=24, xend=24, y=0, yend=10, size=0.3, col=hsv(0,0,0.5), linetype=3)
g <- g + facet_grid(.~cond, labeller=as_labeller(c('b'="T1",'c'="T2",'d'="T3"))) #+ geom_smooth(method="lm")
g <- g + theme_bw() + theme(strip.background=element_blank())
g <- g + xlab("Trial") + ylab("Bid") + scale_color_discrete(labels=c("SC","NC","BC"), name="")
g <- g + scale_x_continuous(breaks=seq(5,24,5))
g <- g + theme(legend.position=c(.07,.2), legend.background=element_blank(), legend.text=element_text(size=6))
print(g)
dev.off()


## Page's trend test of b(3), c(3)
# most often used with fairly small numbers of conditions and subjects
library(crank)
for (i in ml) for (j in nl) {
  ptm <- matrix(h$bid[h$cond==j & h$snb==i], nrow=nlevels(h$sid)/nlevels(h$cond))
  ptt <- page.trend.test(ranks=F, x=ptm)
  str(ptt)
}


## Mean of 40(or 24) differences B-N and N-S in condition c
hp <- h24
meanbids_bn_b <- list(); meanbids_bn_c <- list()
for (i in ml) meanbids_bn_b[[i]] <- by(hp$bid[hp$snb==i&hp$cond=="b"], hp$block_n[hp$snb==i&hp$cond=="b"], function(x) mean(x, na.rm=T))
for (i in ml) meanbids_bn_c[[i]] <- by(hp$bid[hp$snb==i&hp$cond=="c"], hp$block_n[hp$snb==i&hp$cond=="c"], function(x) mean(x, na.rm=T))

# fig06: "Differences between markets of mean bids across trials"
tiff("fig06.tif", width=88, height=60, units="mm", pointsize=8, res=300)
par(mfrow=c(1,2), mar=c(4,4,2,1))
boxplot(meanbids_bn_b$BC - meanbids_bn_b$NC, meanbids_bn_b$SC - meanbids_bn_b$NC, ylim=c(-1,4), ylab="Bid shifts")
axis(1, at=1:2, labels=c("BC-NC", "SC-NC")); title("Correct labels (T1)")
boxplot(meanbids_bn_c$BC - meanbids_bn_c$NC, meanbids_bn_c$SC - meanbids_bn_c$NC, ylim=c(-1,4), ylab="")
axis(1, at=1:2, labels=c("BC-NC", "SC-NC")); title("Scrambled labels (T2) ")
dev.off()


# same as above with barplot
par(mfrow=c(1,2))
diff_BN <- c(mean(meanbids_bn_b$BC - meanbids_bn_b$NC), 1.96*sd(meanbids_bn_b$BC - meanbids_bn_b$NC)/sqrt(39))
diff_NS <- c(mean(meanbids_bn_b$NC - meanbids_bn_b$SC), 1.96*sd(meanbids_bn_b$NC - meanbids_bn_b$SC)/sqrt(39))
barCenters <- barplot(c(diff_BN[1],diff_NS[1]), ylim=c(0,4), main="correct labels (T1)")
axis(1, at=barCenters, labels=c("BC-NC", "NC-SC"))
arrows(barCenters, c(diff_BN[1]-diff_BN[2],diff_NS[1]-diff_NS[2]), barCenters, c(diff_BN[1]+diff_BN[2],diff_NS[1]+diff_NS[2]), angle=90, code=3)
#segments(barCenters, c(diff_BN[1]-diff_BN[2],diff_NS[1]-diff_NS[2]), barCenters, c(diff_BN[1]+diff_BN[2],diff_NS[1]+diff_NS[2]))
diff_BN <- c(mean(meanbids_bn_c$BC - meanbids_bn_c$NC), 1.96*sd(meanbids_bn_c$BC - meanbids_bn_c$NC)/sqrt(39))
diff_NS <- c(mean(meanbids_bn_c$NC - meanbids_bn_c$SC), 1.96*sd(meanbids_bn_c$NC - meanbids_bn_c$SC)/sqrt(39))
barCenters <- barplot(c(diff_BN[1],diff_NS[1]), ylim=c(0,2), main="scrambled labels (T2)")
axis(1, at=barCenters, labels=c("BC-NC", "NC-SC"))
arrows(barCenters, c(diff_BN[1]-diff_BN[2],diff_NS[1]-diff_NS[2]), barCenters, c(diff_BN[1]+diff_BN[2],diff_NS[1]+diff_NS[2]), angle=90, code=3)


## Mean bids corrected for scrambled label condition using only mean summary statistic
# boxplot 
par(mfrow=c(1,1))
boxplot(meanbids_bn_b$BC - meanbids_bn_c$BC, meanbids_bn_b$NC - meanbids_bn_c$NC, meanbids_bn_b$SC - meanbids_bn_c$SC)

# scatterplot fig06
plot(meanbids_bn_b$BC - meanbids_bn_c$BC, col="red", ylim=c(-3.2,1), pch=0, main="", xlab="Trial", ylab="T1 mean bids - T2 mean bids")
points(meanbids_bn_b$NC - meanbids_bn_c$NC, col="green", pch=1)
points(meanbids_bn_b$SC - meanbids_bn_c$SC, col="blue", pch=2)
#abline(v=24, lwd=1, col=rgb(0,0,0,0.5))
legend("bottomleft", c("SC","NC","BC"), pch=0:2, col=cl1)


## Between subject stats: Histogram of subject transaction rates
par(mar=c(4,4,2,2))
plot(density(xtabs(formula=out_bool~sid,data=hp)), xlab="Accepted transactions out of 72", main="")

hist(tapply(hp[hp$cond=="b",]$out_bool, hp[hp$cond=="b",]$sid, sum), breaks=20, col=rgb(1,0,0,0.3), main="",ylab="Counts",xlab="Accepted transactions out of 72")
for (i in 1:3) {
  df <- by(hp[hp$cond==nl[i],]$out_bool, hp[hp$cond==nl[i],]$sid, sum)
  df <- df[complete.cases(df)]
  if  (i==1) {
    hist(df, breaks=30, col=cl0[i], main="",ylab="",xlab="Accepted transactions out of 120")
    legend("topright", c("BC","NC","SC"), pch=c(15), col=cl0)
  } else hist(df, col=cl0[i], breaks=30, add=T)
}
hist(xtabs(formula=out_bool~sid,data=h2), col=rgb(1,1,1,0.3), breaks=30, add=T) 

# fig07
tiff("fig07.tif", width=88, height=60, units="mm", pointsize=8, res=300)
par(mfrow=c(1,1), mar=c(4,4,2,1))
for (i in 1:3) {
  df <- by(hp[hp$cond==nl[i],]$out_bool, hp[hp$cond==nl[i],]$sid, sum)
  df <- df[complete.cases(df)]
  if  (i==1) {
    plot(density(df), col=cl1[i], xlim=c(20,72), main="",ylab="Arbitrary units", ylim=c(0,0.1), yaxt="n", xlab="Number of accepted transactions (max. 72)")
    legend("topright", condls, pch=c(15), col=cl1)
  } else lines(density(df), col=cl1[i])
} 
#lines(density(xtabs(formula=out_bool~sid,data=hp)), col="black") 
dev.off()


## Seller behavior
opp_types = c("seller_NC","seller_SC","buyer_BC","seller_BC")
h3 <- read.csv(paste0(cwd, "/PrerecordedData.csv"), header=F)
h3$opp_role <- gl(4,24, labels=opp_types)
h3$nblock <- factor(rep(1:24, times=4))
h3 <- reshape(h3, varying = names(h3)[-c(17:18)], v.names="bid", timevar="opp_rand", direction="long", idvar="previd")
h3$bid[h3$bid == 100] <- NA

# Fill missing blocks
#h3 <- h3[complete.cases(h3),] instead of this
for (i in which(!complete.cases(h3))) h3[i,"bid"] <- mean(h3[c(i-1,i+1),"bid"])

# opponents bids and ask prices distributions
bwplot(bid ~ nblock | opp_role, data=h3)
table(h3$opp_role, h3$nblock)
boxplot(bid ~ opp_role, data=h3)
densityplot(~bid, data=h3,groups=opp_role,par.settings=list(superpose.line=list(col=cl1)),auto.key=list(text=opp_types, corner=c(0,1)), col=cl1, xlab="MU", main="")

hist(h3[h3$opp_role=="seller_SC",]$bid, breaks=50, col=rgb(1,0,0,0.3), main="",ylab="Counts",xlab="MU")
hist(h3[h3$opp_role=="seller_NC",]$bid, breaks=50, col=rgb(0,1,0,0.3), add=T)
hist(h3[h3$opp_role=="seller_BC",]$bid, breaks=50, col=rgb(0,0,1,0.3), add=T)
hist(h3[h3$opp_role=="buyer_BC",]$bid, breaks=50, col=rgb(0.3,0.3,0.3,0.3), add=T)


## Parametric density estimation
plot(dbeta(seq(0,1,0.01), 7, 12))
plot(dbinom(seq(0,1,0.01), 7, 12))
plot(dbinom(seq(0,40), 40, 0.3))
plot(dhyper(seq(0,100), 70, 30, 50))


## Kernel density estimation of opponent behavior pdf
bw <- plot(bkde(h3$bid, kernel="normal", bandwidth=0.28, gridsize=101, range=c(0,10)))
bkde(h3$bid, kernel="normal", bandwidth=bw, gridsize=101, range=c(0,10))
plot(density(h3$bid, bw="SJ", kernel="gaussian", adjust=1, weights=NULL, n=101, from=0, to=10, na.rm=T))
plot(ksmooth(seq(0,10,0.1), h3$bid, kernel="normal", bandwidth=0.5, range.x=c(0,10), 101, x.points=seq(0,10,0.1)))

# fig02
tiff("fig02.tif", width=88, height=80, units="mm", pointsize=8, res=300)
par(mfrow=c(2,2), mar=c(4,4,1,1))
boxplot(split(h3[h3$opp_role=="buyer_BC",]$bid, h3[h3$opp_role=="buyer_BC",]$nblock), varwidth=T, notch=F, ylim=c(0,10), ylab="Buyer bids in BC")
boxplot(split(h3[h3$opp_role=="seller_BC",]$bid, h3[h3$opp_role=="seller_BC",]$nblock), varwidth=T, notch=F, ylim=c(0,10), ylab="Seller ask prices in BC")
boxplot(split(h3[h3$opp_role=="seller_NC",]$bid, h3[h3$opp_role=="seller_NC",]$nblock), varwidth=T, notch=F, ylim=c(0,10), xlab="Trial number", ylab="Seller ask prices in NC")
boxplot(split(h3[h3$opp_role=="seller_SC",]$bid, h3[h3$opp_role=="seller_SC",]$nblock), varwidth=T, notch=F, ylim=c(0,10), xlab="Trial number", ylab="Seller ask prices in SC")
dev.off()

# fig12
tiff("fig12.tif", width=80, height=90, units="mm", pointsize=8, res=300)
par(mfrow = c(3,1), mar=c(4,4,2,1))
for (i in nl) {
  hist(hp[hp$cond==i,]$bid, 
       xlim=c(0,10), col=rgb(1,1,0,0.5), breaks=50, ylab=paste0("Counts (", condls[[i]], ")"), 
       xlab=list(d="Bid")[[i]], main="") 
}
dev.off()

wilcox.test(hp[hp$cond=="b",]$bid, hp[hp$cond=="c",]$bid)
wilcox.test(hp[hp$cond=="b",]$bid, hp[hp$cond=="d",]$bid)
wilcox.test(hp[hp$cond=="c",]$bid, hp[hp$cond=="d",]$bid)


## fig08: Profit across experiments
tiff("fig08.tif", width=88, height=70, units="mm", pointsize=8, res=300)
par(mfrow=c(1,1))
boxplot(split(hp$profit, hp$cond), varwidth=T, notch=T, xaxt='n', ylab="Bid")
axis(side=1, at=1:3, labels=condls)
dev.off() 

pft_cond2 <- tapply(hp$profit, hp$cond, mean, na.rm=T)
xtabs(profit~sid+cond, hp)
pft_mn <- tapply(hp$profit, hp[,c("sid","cond")], mean, na.rm=T)
kwt <- kruskal.test(formula = profit ~ cond, data = hp)
wilcox.test(pft_mn[1:18,1],pft_mn[19:36,2],paired=F, alternative="greater")
wilcox.test(pft_mn[37:54,3], pft_mn[19:36,2], paired=F, alternative="greater")
wilcox.test(pft_mn[37:54,3],pft_mn[1:18,1],paired=F, alternative="greater")

# Tukey's HSD
glhtf <- glht(avf, linfct = mcp(Frequency="Tukey", interaction_average = F), alternative="two.sided")
# see glhtf$linfct
summary(glhtf, test=adjusted(type="holm"))



# overlapping histograms
hist(h[h$cond=="b",]$profit, breaks=50, col=rgb(1,0,0,0.3), main="",ylab="Counts",xlab="MU")
hist(h[h$cond=="c",]$profit, breaks=50, col=rgb(0,1,0,0.3), add=T)
hist(h[h$cond=="d",]$profit, breaks=50, col=rgb(0,0,1,0.3), add=T)

histogram(h$profit, breaks=50, ylab="Counts",xlab="MU", main="", groups=h$cond,
          panel = function(...) panel.superpose(..., panel.groups=panel.histogram, col=cl1, alpha=0.4),
          auto.key=list(corner=c(1,1), text=condls))

# overlapping density plots. Smoothed profit histogram
densityplot(~profit, data=h, groups=cond, 
            par.settings=list(superpose.line=list(col=cl1)),
            auto.key=list(text=condls, corner=c(1,1)), 
            col=cl1, xlab="MU", main="")

## fig09: BidVariation histogram by PrevOutcome 
hp <-h24
tiff("fig09.tif", width=88, height=100, units="mm", pointsize=8, res=300)
par(mfrow = c(3,1), mar=c(4,4,2,1))
for (i in nl) {
  hist(hp[as.logical(hp$out_prev) & hp$cond==i,]$bid - hp[as.logical(hp$out_prev) & hp$cond==i,]$bid_prev, 
       xlim=c(-3.5,3.5), col=rgb(1,1,0,0.5), breaks=100, ylab=paste0("Counts (", condls[[i]], ")"), 
       xlab=list(d="Bid shifts")[[i]], main="") #"Bid adjustment counts by previous bid acceptance contingency")
  hist(hp[!hp$out_prev & hp$cond==i,]$bid - hp[!hp$out_prev & hp$cond==i,]$bid_prev, col=rgb(1,0,1,0.5), breaks=100, add=T)
}
legend("right", bty="n", pt.bg=c(rgb(1,1,0,0.5),rgb(1,0,1,0.5)), c("Previous bid accepted"," Previous bid rejected"), 
       pch=c(22,22), col=c(rgb(1,1,0,0.5),rgb(1,0,1,0.5)))
dev.off()

prmnsd <- function(x) {print(median(x, na.rm=T)); print(mean(x, na.rm=T)); 
  print(sd(x, na.rm=T)); print(skewness(x, na.rm=T)); print(kurtosis(x, na.rm=T)-3)}
for (i in nl) {
  prmnsd(hp[as.logical(hp$out_prev) & hp$cond==i,]$bid - hp[as.logical(hp$out_prev) & hp$cond==i,]$bid_prev)
  prmnsd(hp[!hp$out_prev & hp$cond==i,]$bid - hp[!hp$out_prev & hp$cond==i,]$bid_prev)  
}
prmnsd(hp[as.logical(hp$out_prev),]$bid - hp[as.logical(hp$out_prev),]$bid_prev)
prmnsd(hp[!hp$out_prev,]$bid - hp[!hp$out_prev,]$bid_prev)


## Model fits
sL <- list()
sL$dr_avf_b <- c(-281.96390175781511, -289.82538564135564, -255.66665526789984, -211.8092869538159, -273.94880158051268, -220.60370082616205, -299.02747840893988, -263.49028893040668, -295.79202026501008, -231.70908882743404, -218.98227986018796, -248.31717950084331, -174.67605836549504, -217.11937004390649, -267.60257019733251, -280.04571520040309, -245.40196342698806, -283.83263319084818)*3/5
sL$dr_avf_c <- c(-225.85571392502791, -228.78640395228578, -189.36881958144613, -183.017812103491, -223.65569243789579, -225.18918909668363, -256.94043687420117, -237.63369930783506, -259.2099911736297, -206.90029497064518, -185.15518362221187, -192.63534554102222, -141.77960816237132, -169.67336900149544, -217.38745441459565, -231.5693641042327, -194.47678444738912, -298.42837713405481)*3/5
sL$dr_avf_d <- c(-178.72231616011368, -189.89818783230663, -163.04531158844577, -151.52316332169164, -191.0533507548291, -176.79365741245107, -191.86003983390285, -217.43139286257988, -211.98801624347192, -151.69383542172685, -159.91943759392427, -158.42293005318385, -118.58718401075063, -160.5277653893823, -174.47748792663202, -187.47547947921285, -170.95849687580554, -241.29980852250776)*3/5

sL$dr_nudg_b <- c(-35.130966616236613, -56.337304032828705, -21.741517045859588, -39.856045318012839, -87.749344858074338, -52.685964147524096, -31.409893482476242, -89.05156154718776, -73.502226868858983, -27.399027482135679, -38.683130465693893, -29.306453919104733, -33.093408265771053, -47.476010698645915, -32.122007245164433, -47.910590788361461, -23.632351224846161, -122.58156029880499)*3/5
sL$dr_nudg_c <- c(-38.760911500589565, -55.253434088480816, -26.540124600048014, -44.198493458241842, -81.900111283256408, -55.755011652027562, -34.890374079516334, -83.912430583912894, -70.715675808563489, -30.442385674401791, -40.425123229645521, -32.561141410333043, -33.524658709351534, -46.613471110083047, -35.913927487083825, -48.782113246863979, -27.962410517645345, -112.88557589332557)*3/5
sL$dr_nudg_d <- c(-35.510597637329163, -56.659835539566508, -22.36058134203455, -40.004762900092238, -89.627223482783222, -50.833758751644595, -32.207870169147462, -87.814869549127934, -74.177808329312512, -26.571132268528807, -39.637140399659685, -28.748927291862728, -29.708735157891667, -47.024334348880934, -32.267896842412696, -47.541085911331592, -21.651390759706644, -128.40602738455058)*3/5

sL$kde_b <- c(-189.50173495937759, -211.14592691544996, -202.06244405530822, -212.30741255913597, -202.30030866987445, -213.0013721952634, -234.76654643710208, -212.90459076837922, -224.90654320049384, -227.80263359642842, -219.68765028688787, -214.03016429386747, -194.88241472121445, -206.14226090516365, -218.99749014091336, -227.16114732842885, -202.22865753772547, -204.85331763791984)*3/5
sL$kde_c <- c(-202.47737148277773, -237.59567160373228, -209.44312828841234, -206.38306793682213, -208.11228687767951, -201.00532424847214, -251.25880578055748, -209.37043862943426, -212.52119860454937, -241.12215366443894, -257.52040475977788, -221.57594879174019, -200.74913984464121, -244.28377197033771, -214.1885350415279, -211.2511353326737, -202.79070754470507, -212.34947917631644)*3/5
sL$kde_d <- c(-182.02334038829142, -197.32626889816027, -188.45963249682859, -222.97045159682858, -235.2593066777174, -172.72663420136115, -188.56242840101223, -246.40098332842297, -201.86694997572698, -189.64083570882357, -234.26844677638343, -251.40583931523793, -193.04554064562564, -190.99115044241566, -190.80423606182455, -176.62580084686766, -188.99015934943216, -185.49259954459646)*3/5


modselFn2 <- "~/MEGA/Neuroscience/Emotion_Action_Control_Learning/Neuroeconomics/pjShyuka/behmod/BicModelSelection_Econo.txt"
ms <- read.table(modselFn2, sep="", col.names=c("BIC","CAIC","AgentName","V4"))
ms[[4]] <- NULL
ms$CAIC <- sapply(ms$CAIC, function(x) as.numeric(gsub("[[(,]","", x)))
ms$BIC <- sapply(ms$BIC, function(x) as.numeric(gsub("[[(,]","", x)))
agmods <- c("null_b", "null_c", "null_d", "avf_101_b", "avf_101_c", "avf_101_d", 
            "lepkurnudger6_b", "lepkurnudger6_c", "lepkurnudger6_d",
            "lepkurnudger7_b", "lepkurnudger7_c", "lepkurnudger7_d", "kde_b", "kde_c", "kde_d")
agmods2 <- c("null_b", "null_c", "null_d", "avf_b", "avf_c", "avf_d", 
             "lkn6_b", "lkn6_c", "lkn6_d", "lkn7_b", "lkn7_c", "lkn7_d", "kde_b", "kde_c", "kde_d")
agmods3 <- c("null", "avf", "lkn6", "lkn7", "kde")
agl2 <- mapply(function(x) grepl(x, ms$AgentName), agmods)
ms$agmod <- factor(apply(agl2,1,function(x) names(which(x))), agmods)
ms <- ms[ms$AgentName != "11111_ffx_dr_lepkurnudger32_b",]
ms$agmod3 <- factor(c("lkn7", "lkn6", "lkn7", "lkn6", "lkn7", "lkn6", "lkn6", rep("avf",3), rep("kde",3), rep("null",3)))
ms <- ms[ms$agmod3 != "lkn6",]

barchart(ms$BIC ~ ms$agmod, ylab="BIC score", xlab="Model type", main="BIC scores across conditions and models")
#barchart(x ~ Group.1, aggregate(ms$BIC, by=list(ms$agmod), mean))
#stripplot(ms$BIC ~ ms$agmod, ylab="BIC score", xlab="Model type", main="BIC scores across conditions and models")
boxplot(ms$BIC ~ ms$agmod, notch=T, varwidth=T, main="Model fits comparison by class", ylab="BIC score")
bwplot(ms$BIC ~ ms$agmod, ylab="BIC score", main="Model fits comparison by class") #barchart(ms$BIC ~ ms$agmod)
dotplot(ms$BIC ~ ms$agmod, ylab="BIC score", main="Model fits comparison by class")

palette(rainbow(3));
palette(); levels(ms$agmod)
par(mar=c(2,4,2,2), mfrow=c(1,1))
{barplot(sort(ms$BIC, decreasing=T), col=sort(ms$agmod), ylab="BIC score", main="Model fits comparison")
  #legend(x="topright", levels(ms$agmod), pch=c(22,22), col=palette())
  legend(x="topright", c("T1 Null", "T2 Null", "T3 Null", "T1 RL", "T2 RL", "T3 RL", "T1 DL", "T2 DL", "T3 DL"), pch=c(22,22), col=palette())}

# calculate from BicModelSelection2_sL confidence intervals with mean(v), sd(em)*1.96
bic <- function(L,n,m) -2*L + n*log10(m)
sem <- function(x) sqrt(var(x,na.rm=T)/length(na.omit(x)))
null_sL <- -4329.3/18
ms$BIC_sLmn <- rep(0,12); ms$BIC_sLci <- rep(0,12)
ms$BIC_sLmn[5] <- mean(mapply(bic, sL$dr_avf_b, 9, 120)); ms$BIC_sLci[5] <- 1.96*sem(mapply(bic, sL$dr_avf_b, 9, 120))
ms$BIC_sLmn[6] <- mean(mapply(bic, sL$dr_avf_c, 9, 120)); ms$BIC_sLci[6] <- 1.96*sem(mapply(bic, sL$dr_avf_c, 9, 120));
ms$BIC_sLmn[4] <- mean(mapply(bic, sL$dr_avf_d, 9, 120)); ms$BIC_sLci[4] <- 1.96*sem(mapply(bic, sL$dr_avf_d, 9, 120))
ms$BIC_sLmn[1] <- mean(mapply(bic, sL$dr_nudg_b, 5, 120)); ms$BIC_sLci[1] <- 1.96*sem(mapply(bic, sL$dr_nudg_b, 5, 120))
ms$BIC_sLmn[3] <- mean(mapply(bic, sL$dr_nudg_c, 5, 120)); ms$BIC_sLci[3] <- 1.96*sem(mapply(bic, sL$dr_nudg_c, 5, 120))
ms$BIC_sLmn[2] <- mean(mapply(bic, sL$dr_nudg_d, 5, 120)); ms$BIC_sLci[2] <- 1.96*sem(mapply(bic, sL$dr_nudg_d, 5, 120))
ms$BIC_sLmn[8] <- mean(mapply(bic, sL$kde_b, 1, 120)); ms$BIC_sLci[8] <- 1.96*sem(mapply(bic, sL$kde_b, 1, 120))
ms$BIC_sLmn[9] <- mean(mapply(bic, sL$kde_c, 1, 120)); ms$BIC_sLci[9] <- 1.96*sem(mapply(bic, sL$kde_c, 1, 120))
ms$BIC_sLmn[7] <- mean(mapply(bic, sL$kde_d, 1, 120)); ms$BIC_sLci[7] <- 1.96*sem(mapply(bic, sL$kde_d, 1, 120))
ms$BIC_sLmn[10:12] <- mapply(bic, rep(null_sL, 18), 120, 120); ms$BIC_sLci[7:9] <- 0

# fig10: averaged across subjects sL 
ms <- ms[order(ms$agmod),]
ms$BIC_sLmn <- ms$BIC_sLmn*3/5 # if 24
tiff("fig10.tif", width=88, height=77, units="mm", pointsize=8, res=300)
{bc<-barplot(ms$BIC_sLmn, col=ms$agmod, ylab="BIC score", main="Goodness of fit")
  axis(1, at=bc[c(2,5,8,11)], labels=c("Null","RL","DL","BDEL"), tick=F)
  legend(x="topright", c("T1", "T2", "T3"), pch=c(15,15), col=palette())
  segments(bc, ms$BIC_sLmn+ms$BIC_sLci, bc, ms$BIC_sLmn-ms$BIC_sLci, lwd=1)
  arrows(bc, ms$BIC_sLmn+ms$BIC_sLci, bc, ms$BIC_sLmn-ms$BIC_sLci, lwd=1, angle=90, code=3, length=0.06)}
dev.off()

# BIC scores across conditions and models
barchart(BIC ~ agmod, groups=agmod3, data=ms, ylab="BIC score", xlab="Model type", main="", scales=list(x=list(rot=90,cex=0.8)))


### Simulation and behavior plotting
cwd3 <- '~/MEGA/Neuroscience/Emotion_Action_Control_Learning/Neuroeconomics/pjShyuka/behmod/'
sv <- list()
bv <- list()
for (i in dir(cwd3)) { 
  if (grepl("econo_simvars_11111_ffx_dr_lepkurnudger6.*", i)) {
    bcd <- regmatches(i, regexec('econo_simvars_11111_ffx_dr_lepkurnudger6.*([a-z])\\.csv$', i))[[1]][2]
    s <- read.csv(paste0(cwd3, i), header=TRUE)
    s$cond <- rep(bcd, nrow(s))
    sv <- rbind(sv, s) 
  } else if (grepl("econo_behvars_drlepkurnudger6.*", i)) {
    bcd <- regmatches(i, regexec('econo_behvars_drlepkurnudger6.*([a-z])\\.csv$', i))[[1]][2]
    b <- read.csv(paste0(cwd3, i), header=TRUE)
    b$cond <- rep(bcd, nrow(b))
    bv <- rbind(bv, b) } 
} 
bv$cond <- factor(bv$cond); sv$cond <- factor(sv$cond)
names(sv)<-c("s_beh", "s_opb", "s_clh", "s_rpe", "s_cob", "s_coe", "s_cmr", "s_otb", "s_sib")
names(bv)<-c("b_opb", "b_clh", "b_rpe", "b_cob", "b_coe", "b_cmr", "b_sve")

hsb <- cbind(h, sv, bv)
hp3 <- hsb[hsb$block_n < 25,]

# fig11
tiff("fig11.tif", width=99, height=70, units="mm", pointsize=8, res=300)
g <- ggplot(data = hp3, mapping = aes(x=block_n, y=s_opb, color=snb))
g <- g + geom_point(alpha=0.2, size=.5) + geom_smooth(method="loess", se=TRUE, level=0.95)
g <- g + facet_grid(.~cond, labeller=as_labeller(c('b'="T1",'c'="T2",'d'="T3"))) 
#g <- g + geom_segment(x=24, xend=24, y=0, yend=10, size=0.3, col=hsv(0,0,0.5), linetype=3)
g <- g + theme_bw() + theme(strip.background=element_blank())
g <- g + xlab("Trial") + ylab("Bid") + scale_color_discrete(labels=c("SC","NC","BC"), name="")
g <- g + scale_x_continuous(breaks=seq(5,24,5))
g <- g + theme(legend.position=c(.07,.2), legend.background=element_blank(), legend.text=element_text(size=6))
print(g)
dev.off()

#g <- g + geom_smooth(color="#B0B0B0", method="loess", se=TRUE, level=0.95)
#g <- g + geom_smooth(data=hpg, aes(y=bid, color="#808080"), method="loess", se=TRUE, level=0.95)
#g <- g + geom_segment(x=24, xend=24, y=0, yend=10, size=0.3, col=hsv(0,0,0.5), linetype=3)
dev.off()



# Same plotting for behavioral data

#accrej
par(mfrow = c(1,1), mar=c(4,4,2,1))
boxplot(split(h$s_opb[h$snb=="SC"&h$out_bool], h$block_n[h$snb=="SC"&h$out_bool]), col=rgb(1,0,0,0.9), ylim= c(0,10), ylab="", xlab="", notch=T, varwidth=T)
boxplot(split(h$s_opb[h$snb=="NC"&h$out_bool], h$block_n[h$snb=="NC"&h$out_bool]), col=rgb(0,1,0,0.9), notch=T, varwidth=T, add=T)
boxplot(split(h$s_opb[h$snb=="BC"&h$out_bool], h$block_n[h$snb=="BC"&h$out_bool]), col=rgb(0,0,1,0.9), notch=T, varwidth=T, add=T)
bid_sr_split <- split(h$s_opb[h$snb=="SC"&!h$out_bool], h$block_n[h$snb=="SC"&!h$out_bool])
nbsrs <- as.list(rep(mean(h$s_opb[h$snb=="SC"&!h$out_bool], na.rm=T),24)) #names(nbsrs) <- as.character(1:24)
for (i in 1:24) nbsrs[i] <- bid_sr_split[as.character(i)]
boxplot(nbsrs, col=rgb(1,0,0,0.2), notch=T, varwidth=T, xlim=c(1,24), add=T)
boxplot(split(h$s_opb[h$snb=="NC"&!h$out_bool], h$block_n[h$snb=="NC"&!h$out_bool]), col=rgb(0,1,0,0.2), notch=T, varwidth=T, add=T)
boxplot(split(h$s_opb[h$snb=="BC"&!h$out_bool], h$block_n[h$snb=="BC"&!h$out_bool]), col=rgb(0,0,1,0.2), notch=T, varwidth=T, add=T)
legend("bottom", bty="n", border="#00000020", ncol=3, x.intersp=0.5, col=c("#0000FFE6","#0000FF33","#00FF00E6","#00FF0033","#FF0000E6","#FF000033"),
       c("SC, Bid accepted","SC, Bid rejected","NC, Bid accepted","NC, Bid rejected","BC, Bid accepted","BC, Bid rejected"), pch=c(16,16))
mtext("Trial number", side=1, line=3, cex=1.4)
mtext("Bid", side=2, line=2, cex=1.4)
#accrej collapsed
par(mfrow = c(1,1), mar=c(4,4,2,1))
boxplot(split(h$s_opb[h$snb=="SC"], h$block_n[h$snb=="SC"]), col=rgb(1,0,0,0.3), ylim= c(0,10), ylab="", xlab="", notch=T, varwidth=T, whisklty=0, staplelty=0, outline=F)
boxplot(split(h$s_opb[h$snb=="NC"], h$block_n[h$snb=="NC"]), col=rgb(0,1,0,0.3), add=T, notch=T, whisklty=0, staplelty=0, outline=F)
boxplot(split(h$s_opb[h$snb=="BC"], h$block_n[h$snb=="BC"]), col=rgb(0,0,1,0.3), add=T, notch=T, whisklty=0, staplelty=0, outline=F)
legend("bottom", bty="n", border="#00000020", ncol=3, x.intersp=0.5, col=c("#0000FFE6","#0000FF33","#00FF00E6"),
       c("SC","NC","BC"), pch=c(15,15))
mtext("Trial number", side=1, line=3, cex=1.4)
mtext("Bid", side=2, line=2, cex=1.4)



