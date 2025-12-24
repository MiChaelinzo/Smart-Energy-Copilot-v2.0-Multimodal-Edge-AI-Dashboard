import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Fullscreen as FullscreenIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts';
import { format, parseISO } from 'date-fns';
import { useDashboard } from '../../contexts/DashboardContext';

type ChartType = 'line' | 'area' | 'bar';
type TimeRange = '24h' | '7d' | '30d';

const ConsumptionChartWidget: React.FC = () => {
  const { state } = useDashboard();
  const [chartType, setChartType] = useState<ChartType>('line');
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');

  // Process data for chart
  const processChartData = () => {
    if (state.energyData.length === 0) return [];

    // Filter data based on time range
    const now = new Date();
    const cutoffTime = new Date();
    
    switch (timeRange) {
      case '24h':
        cutoffTime.setHours(now.getHours() - 24);
        break;
      case '7d':
        cutoffTime.setDate(now.getDate() - 7);
        break;
      case '30d':
        cutoffTime.setDate(now.getDate() - 30);
        break;
    }

    const filteredData = state.energyData
      .filter(item => new Date(item.timestamp) >= cutoffTime)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    // Group data by appropriate intervals
    const groupedData = groupDataByInterval(filteredData, timeRange);

    return groupedData.map(item => ({
      ...item,
      formattedTime: formatTimeForDisplay(item.timestamp, timeRange),
    }));
  };

  const groupDataByInterval = (data: any[], range: TimeRange) => {
    if (range === '24h') {
      // Group by hour
      const hourlyData: { [key: string]: any } = {};
      
      data.forEach(item => {
        const hour = format(parseISO(item.timestamp), 'yyyy-MM-dd HH:00:00');
        if (!hourlyData[hour]) {
          hourlyData[hour] = {
            timestamp: hour,
            consumption_kwh: 0,
            cost_usd: 0,
            count: 0,
          };
        }
        hourlyData[hour].consumption_kwh += item.consumption_kwh;
        hourlyData[hour].cost_usd += item.cost_usd;
        hourlyData[hour].count += 1;
      });

      return Object.values(hourlyData).map(item => ({
        ...item,
        consumption_kwh: item.consumption_kwh / item.count,
        cost_usd: item.cost_usd / item.count,
      }));
    } else {
      // Group by day for 7d and 30d
      const dailyData: { [key: string]: any } = {};
      
      data.forEach(item => {
        const day = format(parseISO(item.timestamp), 'yyyy-MM-dd');
        if (!dailyData[day]) {
          dailyData[day] = {
            timestamp: day,
            consumption_kwh: 0,
            cost_usd: 0,
            count: 0,
          };
        }
        dailyData[day].consumption_kwh += item.consumption_kwh;
        dailyData[day].cost_usd += item.cost_usd;
        dailyData[day].count += 1;
      });

      return Object.values(dailyData);
    }
  };

  const formatTimeForDisplay = (timestamp: string, range: TimeRange) => {
    const date = parseISO(timestamp);
    
    switch (range) {
      case '24h':
        return format(date, 'HH:mm');
      case '7d':
        return format(date, 'MMM dd');
      case '30d':
        return format(date, 'MM/dd');
      default:
        return format(date, 'HH:mm');
    }
  };

  const chartData = processChartData();

  const handleChartTypeChange = (event: React.MouseEvent<HTMLElement>, newType: ChartType) => {
    if (newType !== null) {
      setChartType(newType);
    }
  };

  const handleTimeRangeChange = (event: React.MouseEvent<HTMLElement>, newRange: TimeRange) => {
    if (newRange !== null) {
      setTimeRange(newRange);
    }
  };

  const renderChart = () => {
    const commonProps = {
      data: chartData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (chartType) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="formattedTime" />
            <YAxis />
            <RechartsTooltip
              labelFormatter={(label) => `Time: ${label}`}
              formatter={(value: number, name: string) => [
                name === 'consumption_kwh' ? `${value.toFixed(2)} kWh` : `$${value.toFixed(2)}`,
                name === 'consumption_kwh' ? 'Consumption' : 'Cost'
              ]}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="consumption_kwh"
              stackId="1"
              stroke="#1976d2"
              fill="#1976d2"
              fillOpacity={0.6}
              name="Consumption (kWh)"
            />
          </AreaChart>
        );
      
      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="formattedTime" />
            <YAxis />
            <RechartsTooltip
              labelFormatter={(label) => `Time: ${label}`}
              formatter={(value: number, name: string) => [
                name === 'consumption_kwh' ? `${value.toFixed(2)} kWh` : `$${value.toFixed(2)}`,
                name === 'consumption_kwh' ? 'Consumption' : 'Cost'
              ]}
            />
            <Legend />
            <Bar dataKey="consumption_kwh" fill="#1976d2" name="Consumption (kWh)" />
          </BarChart>
        );
      
      default: // line
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="formattedTime" />
            <YAxis />
            <RechartsTooltip
              labelFormatter={(label) => `Time: ${label}`}
              formatter={(value: number, name: string) => [
                name === 'consumption_kwh' ? `${value.toFixed(2)} kWh` : `$${value.toFixed(2)}`,
                name === 'consumption_kwh' ? 'Consumption' : 'Cost'
              ]}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="consumption_kwh"
              stroke="#1976d2"
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
              name="Consumption (kWh)"
            />
            <Line
              type="monotone"
              dataKey="cost_usd"
              stroke="#dc004e"
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
              name="Cost (USD)"
            />
          </LineChart>
        );
    }
  };

  return (
    <Card className="widget-container" sx={{ height: '100%' }}>
      <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Widget Header */}
        <Box className="widget-header">
          <Typography variant="h6" component="h2">
            Energy Consumption Trends
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Zoom In">
              <IconButton size="small">
                <ZoomInIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Zoom Out">
              <IconButton size="small">
                <ZoomOutIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Fullscreen">
              <IconButton size="small">
                <FullscreenIcon />
              </IconButton>
            </Tooltip>
            <IconButton size="small">
              <MoreVertIcon />
            </IconButton>
          </Box>
        </Box>

        {/* Controls */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2, flexWrap: 'wrap', gap: 1 }}>
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={handleChartTypeChange}
            size="small"
          >
            <ToggleButton value="line">Line</ToggleButton>
            <ToggleButton value="area">Area</ToggleButton>
            <ToggleButton value="bar">Bar</ToggleButton>
          </ToggleButtonGroup>

          <ToggleButtonGroup
            value={timeRange}
            exclusive
            onChange={handleTimeRangeChange}
            size="small"
          >
            <ToggleButton value="24h">24H</ToggleButton>
            <ToggleButton value="7d">7D</ToggleButton>
            <ToggleButton value="30d">30D</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {/* Chart */}
        <Box sx={{ flex: 1, minHeight: 0 }}>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              {renderChart()}
            </ResponsiveContainer>
          ) : (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: 'text.secondary',
              }}
            >
              <Typography variant="body1">
                {state.loading ? 'Loading chart data...' : 'No data available for the selected time range'}
              </Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConsumptionChartWidget;