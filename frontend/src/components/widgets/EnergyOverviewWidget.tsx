import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  LinearProgress,
  IconButton,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  MoreVert as MoreVertIcon,
  Bolt as BoltIcon,
  AttachMoney as AttachMoneyIcon,
} from '@mui/icons-material';
import { useDashboard } from '../../contexts/DashboardContext';

const EnergyOverviewWidget: React.FC = () => {
  const { state } = useDashboard();

  // Calculate summary statistics
  const calculateSummary = () => {
    if (state.energyData.length === 0) {
      return {
        totalConsumption: 0,
        totalCost: 0,
        averageEfficiency: 0,
        trend: 'stable' as const,
        trendPercentage: 0,
      };
    }

    const recentData = state.energyData.slice(0, 24); // Last 24 hours
    const previousData = state.energyData.slice(24, 48); // Previous 24 hours

    const totalConsumption = recentData.reduce((sum, item) => sum + item.consumption_kwh, 0);
    const totalCost = recentData.reduce((sum, item) => sum + item.cost_usd, 0);
    const averageEfficiency = recentData.reduce((sum, item) => sum + item.confidence_score, 0) / recentData.length;

    // Calculate trend
    const previousConsumption = previousData.reduce((sum, item) => sum + item.consumption_kwh, 0);
    const trendPercentage = previousConsumption > 0 
      ? ((totalConsumption - previousConsumption) / previousConsumption) * 100 
      : 0;
    
    const trend = trendPercentage > 5 ? 'increasing' : trendPercentage < -5 ? 'decreasing' : 'stable';

    return {
      totalConsumption,
      totalCost,
      averageEfficiency,
      trend,
      trendPercentage: Math.abs(trendPercentage),
    };
  };

  const summary = calculateSummary();

  const getTrendIcon = () => {
    switch (summary.trend) {
      case 'increasing':
        return <TrendingUpIcon color="error" />;
      case 'decreasing':
        return <TrendingDownIcon color="success" />;
      default:
        return <TrendingUpIcon color="disabled" />;
    }
  };

  const getTrendColor = () => {
    switch (summary.trend) {
      case 'increasing':
        return 'error';
      case 'decreasing':
        return 'success';
      default:
        return 'default';
    }
  };

  return (
    <Card className="widget-container" sx={{ height: '100%' }}>
      <CardContent>
        {/* Widget Header */}
        <Box className="widget-header">
          <Typography variant="h6" component="h2">
            Energy Overview
          </Typography>
          <IconButton size="small">
            <MoreVertIcon />
          </IconButton>
        </Box>

        {/* Widget Content */}
        <Box className="widget-content">
          <Grid container spacing={2}>
            {/* Total Consumption */}
            <Grid item xs={12} sm={6}>
              <Box sx={{ textAlign: 'center', p: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                  <BoltIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    Total Consumption
                  </Typography>
                </Box>
                <Typography variant="h4" component="div" color="primary">
                  {summary.totalConsumption.toFixed(1)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  kWh (24h)
                </Typography>
              </Box>
            </Grid>

            {/* Total Cost */}
            <Grid item xs={12} sm={6}>
              <Box sx={{ textAlign: 'center', p: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                  <AttachMoneyIcon color="secondary" sx={{ mr: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    Total Cost
                  </Typography>
                </Box>
                <Typography variant="h4" component="div" color="secondary">
                  ${summary.totalCost.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  USD (24h)
                </Typography>
              </Box>
            </Grid>

            {/* Trend Indicator */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 2 }}>
                {getTrendIcon()}
                <Chip
                  label={`${summary.trend} ${summary.trendPercentage.toFixed(1)}%`}
                  color={getTrendColor()}
                  variant="outlined"
                  size="small"
                  sx={{ ml: 1, textTransform: 'capitalize' }}
                />
              </Box>
            </Grid>

            {/* Efficiency Indicator */}
            <Grid item xs={12}>
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Data Quality
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {(summary.averageEfficiency * 100).toFixed(0)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={summary.averageEfficiency * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </Grid>
          </Grid>

          {/* Status Indicators */}
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-around' }}>
            <Chip
              label={`${state.energyData.length} Records`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={state.loading ? 'Updating...' : 'Live'}
              size="small"
              color={state.loading ? 'warning' : 'success'}
              variant="outlined"
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default EnergyOverviewWidget;