import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  IconButton,
  Grid,
  Chip,
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AttachMoney as AttachMoneyIcon,
  Savings as SavingsIcon,
} from '@mui/icons-material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
} from 'recharts';
import { useDashboard } from '../../contexts/DashboardContext';

type ViewMode = 'breakdown' | 'trends' | 'savings';

const CostAnalysisWidget: React.FC = () => {
  const { state } = useDashboard();
  const [viewMode, setViewMode] = useState<ViewMode>('breakdown');

  // Calculate cost analysis data
  const calculateCostAnalysis = () => {
    if (state.energyData.length === 0) {
      return {
        totalCost: 0,
        averageCost: 0,
        costTrend: 'stable' as const,
        trendPercentage: 0,
        breakdown: [],
        monthlyCosts: [],
        potentialSavings: 0,
      };
    }

    const recentData = state.energyData.slice(0, 30); // Last 30 records
    const previousData = state.energyData.slice(30, 60); // Previous 30 records

    const totalCost = recentData.reduce((sum, item) => sum + item.cost_usd, 0);
    const averageCost = totalCost / recentData.length;

    // Calculate trend
    const previousCost = previousData.reduce((sum, item) => sum + item.cost_usd, 0);
    const trendPercentage = previousCost > 0 
      ? ((totalCost - previousCost) / previousCost) * 100 
      : 0;
    
    const costTrend = trendPercentage > 5 ? 'increasing' : trendPercentage < -5 ? 'decreasing' : 'stable';

    // Cost breakdown by source
    const sourceBreakdown: { [key: string]: number } = {};
    recentData.forEach(item => {
      sourceBreakdown[item.source] = (sourceBreakdown[item.source] || 0) + item.cost_usd;
    });

    const breakdown = Object.entries(sourceBreakdown).map(([source, cost]) => ({
      name: source.replace('_', ' ').toUpperCase(),
      value: cost,
      percentage: (cost / totalCost) * 100,
    }));

    // Monthly cost trends (mock data for demonstration)
    const monthlyCosts = [
      { month: 'Jan', cost: 120, budget: 150 },
      { month: 'Feb', cost: 135, budget: 150 },
      { month: 'Mar', cost: 128, budget: 150 },
      { month: 'Apr', cost: 142, budget: 150 },
      { month: 'May', cost: 156, budget: 150 },
      { month: 'Jun', cost: totalCost, budget: 150 },
    ];

    // Calculate potential savings from recommendations
    const potentialSavings = state.recommendations
      .filter(rec => rec.status === 'pending')
      .reduce((sum, rec) => sum + (rec.estimated_savings?.annual_cost_usd || 0), 0);

    return {
      totalCost,
      averageCost,
      costTrend,
      trendPercentage: Math.abs(trendPercentage),
      breakdown,
      monthlyCosts,
      potentialSavings,
    };
  };

  const analysis = calculateCostAnalysis();

  const handleViewModeChange = (event: React.MouseEvent<HTMLElement>, newMode: ViewMode) => {
    if (newMode !== null) {
      setViewMode(newMode);
    }
  };

  const getTrendIcon = () => {
    switch (analysis.costTrend) {
      case 'increasing':
        return <TrendingUpIcon color="error" />;
      case 'decreasing':
        return <TrendingDownIcon color="success" />;
      default:
        return <TrendingUpIcon color="disabled" />;
    }
  };

  const getTrendColor = () => {
    switch (analysis.costTrend) {
      case 'increasing':
        return 'error';
      case 'decreasing':
        return 'success';
      default:
        return 'default';
    }
  };

  // Colors for pie chart
  const COLORS = ['#1976d2', '#dc004e', '#2e7d32', '#ed6c02', '#9c27b0'];

  const renderBreakdownView = () => (
    <Box>
      {analysis.breakdown.length > 0 ? (
        <>
          <Box sx={{ height: 200, mb: 2 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={analysis.breakdown}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {analysis.breakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip
                  formatter={(value: number) => [`$${value.toFixed(2)}`, 'Cost']}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Box>
          
          <Box>
            {analysis.breakdown.map((item, index) => (
              <Box key={item.name} sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">{item.name}</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    ${item.value.toFixed(2)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={item.percentage}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: 'grey.200',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: COLORS[index % COLORS.length],
                    },
                  }}
                />
              </Box>
            ))}
          </Box>
        </>
      ) : (
        <Typography variant="body2" color="text.secondary" textAlign="center">
          No cost data available
        </Typography>
      )}
    </Box>
  );

  const renderTrendsView = () => (
    <Box>
      {analysis.monthlyCosts.length > 0 ? (
        <Box sx={{ height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={analysis.monthlyCosts}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <RechartsTooltip
                formatter={(value: number, name: string) => [
                  `$${value.toFixed(2)}`,
                  name === 'cost' ? 'Actual Cost' : 'Budget'
                ]}
              />
              <Legend />
              <Bar dataKey="cost" fill="#1976d2" name="Actual Cost" />
              <Bar dataKey="budget" fill="#dc004e" name="Budget" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      ) : (
        <Typography variant="body2" color="text.secondary" textAlign="center">
          No trend data available
        </Typography>
      )}
    </Box>
  );

  const renderSavingsView = () => (
    <Box>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Box sx={{ textAlign: 'center', p: 2 }}>
            <SavingsIcon color="success" sx={{ fontSize: 48, mb: 1 }} />
            <Typography variant="h4" color="success.main">
              ${analysis.potentialSavings.toFixed(0)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Potential Annual Savings
            </Typography>
          </Box>
        </Grid>
        
        <Grid item xs={6}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h6" color="primary">
              ${(analysis.potentialSavings / 12).toFixed(0)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Monthly Savings
            </Typography>
          </Box>
        </Grid>
        
        <Grid item xs={6}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h6" color="secondary">
              {analysis.potentialSavings > 0 ? ((analysis.potentialSavings / (analysis.totalCost * 12)) * 100).toFixed(0) : 0}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Cost Reduction
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={12}>
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Savings Progress
            </Typography>
            <LinearProgress
              variant="determinate"
              value={Math.min((analysis.potentialSavings / 1000) * 100, 100)}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
              {state.recommendations.filter(r => r.status === 'pending').length} recommendations available
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );

  return (
    <Card className="widget-container" sx={{ height: '100%' }}>
      <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Widget Header */}
        <Box className="widget-header">
          <Typography variant="h6" component="h2">
            Cost Analysis
          </Typography>
          <IconButton size="small">
            <MoreVertIcon />
          </IconButton>
        </Box>

        {/* Summary Stats */}
        <Box sx={{ mb: 2 }}>
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  ${analysis.totalCost.toFixed(2)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Total Cost
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                {getTrendIcon()}
                <Chip
                  label={`${analysis.costTrend} ${analysis.trendPercentage.toFixed(1)}%`}
                  color={getTrendColor()}
                  variant="outlined"
                  size="small"
                  sx={{ textTransform: 'capitalize' }}
                />
              </Box>
            </Grid>
          </Grid>
        </Box>

        {/* View Mode Toggle */}
        <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={handleViewModeChange}
            size="small"
          >
            <ToggleButton value="breakdown">Breakdown</ToggleButton>
            <ToggleButton value="trends">Trends</ToggleButton>
            <ToggleButton value="savings">Savings</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {/* Widget Content */}
        <Box className="widget-content" sx={{ flex: 1, overflow: 'auto' }}>
          {viewMode === 'breakdown' && renderBreakdownView()}
          {viewMode === 'trends' && renderTrendsView()}
          {viewMode === 'savings' && renderSavingsView()}
        </Box>
      </CardContent>
    </Card>
  );
};

export default CostAnalysisWidget;