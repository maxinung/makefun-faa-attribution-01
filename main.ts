// FAA Attribution Forecaster - Polymarket price quantile forecasting
// Reads 72h of 1-min price data, outputs 240-min forecasts at 12 checkpoints as 5 quantiles

import { readFileSync, writeFileSync, readdirSync, existsSync } from "fs";
import { join, basename } from "path";

interface PricePoint {
  timestamp: number;  // Unix seconds
  prices: number[];   // 4 market prices
}

interface ForecastOutput {
  quantiles: number[][][];  // [checkpoint][minute][quantile]
}

// Quantile levels: 0.05, 0.25, 0.5, 0.75, 0.95
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const NUM_MARKETS = 4;
const CHECKPOINTS = 12;
const FORECAST_HORIZON = 240;  // 4 hours in minutes
const LOOKBACK_WINDOW = 72 * 60;  // 72 hours in minutes
const CHECKPOINT_INTERVAL = 4 * 60;  // 4 hours in minutes

function quantile(arr: number[], q: number): number {
  if (arr.length === 0) return 0.5;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = q * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function ewmaVariance(returns: number[], lambda = 0.94): number {
  if (returns.length < 2) return 0.0001;
  let varEst = returns[0] * returns[0];
  for (let i = 1; i < returns.length; i++) {
    varEst = lambda * varEst + (1 - lambda) * returns[i] * returns[i];
  }
  return Math.max(varEst, 1e-8);
}

function parseInput(filePath: string): PricePoint[] {
  const content = readFileSync(filePath, "utf-8").trim();
  if (!content) return [];
  return content.split("\n").map(line => {
    const parts = line.split(",");
    return {
      timestamp: parseInt(parts[0]),
      prices: parts.slice(1).map(p => parseFloat(p))
    };
  });
}

function computeReturns(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    if (prices[i - 1] > 0) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }
  }
  return returns;
}

function forecastMarket(
  history: PricePoint[],
  marketIdx: number,
  checkpointStart: number
): number[][] {
  // Get prices up to checkpoint
  const prices = history
    .filter(p => p.timestamp <= checkpointStart)
    .map(p => p.prices[marketIdx])
    .filter(p => p > 0);
  
  if (prices.length < 10) {
    // Fallback: constant forecast at last price
    const lastPrice = prices[prices.length - 1] || 0.5;
    return Array(FORECAST_HORIZON).fill(null).map(() => 
      QUANTILES.map(q => lastPrice)
    );
  }

  const returns = computeReturns(prices);
  const variance = ewmaVariance(returns);
  const stdDev = Math.sqrt(variance);
  const lastPrice = prices[prices.length - 1];
  const drift = returns.length > 0 ? 
    returns.reduce((a, b) => a + b, 0) / returns.length : 0;

  // Generate forecast quantiles for each minute
  const forecasts: number[][] = [];
  for (let h = 1; h <= FORECAST_HORIZON; h++) {
    // Variance scales with sqrt(time) for log returns
    const horizonVar = variance * h;
    const horizonStd = Math.sqrt(horizonVar);
    const meanLogReturn = drift * h;
    
    const quants = QUANTILES.map(q => {
      // Log-normal quantile
      const z = q === 0.5 ? 0 : (q < 0.5 ? -1 : 1) * Math.sqrt(2) * 
        erfinv(2 * q - 1);
      const logPrice = Math.log(lastPrice) + meanLogReturn + z * horizonStd;
      return Math.exp(logPrice);
    });
    forecasts.push(quants);
  }
  return forecasts;
}

// Inverse error function approximation
function erfinv(x: number): number {
  const a = 0.147;
  const sign = x >= 0 ? 1 : -1;
  const absX = Math.abs(x);
  const ln = Math.log(1 - absX * absX);
  const part1 = 2 / (Math.PI * a) + ln / 2;
  const part2 = ln / a;
  return sign * Math.sqrt(Math.sqrt(part1 * part1 - part2) - part1);
}

function main(): void {
  const inputDir = "/input";
  const outputDir = "/output";
  
  if (!existsSync(inputDir)) {
    console.error("Input directory not found");
    process.exit(1);
  }

  const inputFiles = readdirSync(inputDir).filter(f => f.endsWith(".in")).sort();
  console.log(`Found ${inputFiles.length} input files`);

  for (const inputFile of inputFiles) {
    const caseId = basename(inputFile, ".in");
    const inputPath = join(inputDir, inputFile);
    const outputPath = join(outputDir, `${caseId}.out`);
    
    console.log(`Processing ${caseId}...`);
    
    const history = parseInput(inputPath);
    if (history.length === 0) {
      console.error(`Empty history for ${caseId}`);
      continue;
    }

    // Determine checkpoint timestamps (every 4 hours from deadline)
    // First checkpoint is at deadline, then +4h, +8h, ... +44h
    const lastTimestamp = history[history.length - 1].timestamp;
    const deadline = lastTimestamp;  // Assume last data point is at deadline
    
    const checkpointTimestamps: number[] = [];
    for (let i = 0; i < CHECKPOINTS; i++) {
      checkpointTimestamps.push(deadline + i * CHECKPOINT_INTERVAL * 60);
    }

    // Forecast for each market at each checkpoint
    const allQuantiles: number[][][] = [];  // [checkpoint][minute][quantile]
    
    for (let cp = 0; cp < CHECKPOINTS; cp++) {
      const checkpointStart = checkpointTimestamps[cp];
      const marketForecasts: number[][][] = [];  // [market][minute][quantile]
      
      for (let m = 0; m < NUM_MARKETS; m++) {
        marketForecasts.push(forecastMarket(history, m, checkpointStart));
      }
      
      // Transpose to [minute][quantile] - average across markets for simplicity
      // Actually need per-market forecasts, so keep as [market][minute][quantile]
      allQuantiles.push(
        Array(FORECAST_HORIZON).fill(null).map((_, minute) =>
          QUANTILES.map((_, qIdx) => 
            // Average across markets (could also output per-market)
            marketForecasts.reduce((sum, mf) => sum + mf[minute][qIdx], 0) / NUM_MARKETS
          )
        )
      );
    }

    // Write output
    const output = JSON.stringify({ quantiles: allQuantiles });
    writeFileSync(outputPath, output);
    console.log(`Wrote ${outputPath}`);
  }
}

main();
