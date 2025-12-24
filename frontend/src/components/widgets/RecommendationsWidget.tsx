import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  EcoIcon,
  AttachMoney as AttachMoneyIcon,
} from '@mui/icons-material';
import { useDashboard } from '../../contexts/DashboardContext';

const RecommendationsWidget: React.FC = () => {
  const { state, updateRecommendationStatus } = useDashboard();
  const [selectedRecommendation, setSelectedRecommendation] = useState<any>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'cost_saving':
        return <AttachMoneyIcon />;
      case 'efficiency':
        return <TrendingUpIcon />;
      case 'environmental':
        return <EcoIcon />;
      default:
        return <InfoIcon />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'implemented':
        return 'success';
      case 'pending':
        return 'warning';
      case 'dismissed':
        return 'default';
      default:
        return 'default';
    }
  };

  const handleRecommendationClick = (recommendation: any) => {
    setSelectedRecommendation(recommendation);
    setDialogOpen(true);
  };

  const handleStatusUpdate = async (id: string, status: string) => {
    await updateRecommendationStatus(id, status);
    setDialogOpen(false);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedRecommendation(null);
  };

  // Filter and sort recommendations
  const pendingRecommendations = state.recommendations
    .filter(rec => rec.status === 'pending')
    .sort((a, b) => {
      // Sort by priority (high > medium > low) then by confidence
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const aPriority = priorityOrder[a.priority.toLowerCase() as keyof typeof priorityOrder] || 0;
      const bPriority = priorityOrder[b.priority.toLowerCase() as keyof typeof priorityOrder] || 0;
      
      if (aPriority !== bPriority) {
        return bPriority - aPriority;
      }
      
      return b.confidence - a.confidence;
    })
    .slice(0, 5); // Show top 5 recommendations

  return (
    <>
      <Card className="widget-container" sx={{ height: '100%' }}>
        <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Widget Header */}
          <Box className="widget-header">
            <Typography variant="h6" component="h2">
              AI Recommendations
            </Typography>
            <IconButton size="small">
              <MoreVertIcon />
            </IconButton>
          </Box>

          {/* Widget Content */}
          <Box className="widget-content" sx={{ flex: 1, overflow: 'auto' }}>
            {pendingRecommendations.length > 0 ? (
              <List dense>
                {pendingRecommendations.map((recommendation, index) => (
                  <ListItem
                    key={recommendation.id}
                    button
                    onClick={() => handleRecommendationClick(recommendation)}
                    sx={{
                      mb: 1,
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      '&:hover': {
                        backgroundColor: 'action.hover',
                      },
                    }}
                  >
                    <Box sx={{ mr: 2 }}>
                      {getTypeIcon(recommendation.type)}
                    </Box>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle2" noWrap>
                            {recommendation.title}
                          </Typography>
                          <Chip
                            label={recommendation.priority}
                            size="small"
                            color={getPriorityColor(recommendation.priority)}
                            variant="outlined"
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary" noWrap>
                            {recommendation.description}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5, gap: 1 }}>
                            <Typography variant="caption" color="success.main">
                              Save ${recommendation.estimated_savings?.annual_cost_usd?.toFixed(0) || 0}/year
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              •
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {(recommendation.confidence * 100).toFixed(0)}% confidence
                            </Typography>
                          </Box>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Tooltip title="View Details">
                        <IconButton edge="end" size="small">
                          <InfoIcon />
                        </IconButton>
                      </Tooltip>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
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
                  {state.loading ? 'Loading recommendations...' : 'No pending recommendations'}
                </Typography>
              </Box>
            )}

            {/* Summary Stats */}
            {state.recommendations.length > 0 && (
              <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                <Typography variant="caption" color="text.secondary" gutterBottom>
                  Total Recommendations: {state.recommendations.length}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip
                    label={`${state.recommendations.filter(r => r.status === 'pending').length} Pending`}
                    size="small"
                    color="warning"
                    variant="outlined"
                  />
                  <Chip
                    label={`${state.recommendations.filter(r => r.status === 'implemented').length} Implemented`}
                    size="small"
                    color="success"
                    variant="outlined"
                  />
                </Box>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Recommendation Details Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        {selectedRecommendation && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {getTypeIcon(selectedRecommendation.type)}
                <Box>
                  <Typography variant="h6">
                    {selectedRecommendation.title}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                    <Chip
                      label={selectedRecommendation.priority}
                      size="small"
                      color={getPriorityColor(selectedRecommendation.priority)}
                    />
                    <Chip
                      label={selectedRecommendation.type.replace('_', ' ')}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={selectedRecommendation.difficulty}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </Box>
              </Box>
            </DialogTitle>

            <DialogContent>
              <Typography variant="body1" paragraph>
                {selectedRecommendation.description}
              </Typography>

              {/* Estimated Savings */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Estimated Impact</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Annual Cost Savings
                      </Typography>
                      <Typography variant="h6" color="success.main">
                        ${selectedRecommendation.estimated_savings?.annual_cost_usd?.toFixed(2) || 0}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Energy Savings
                      </Typography>
                      <Typography variant="h6" color="primary">
                        {selectedRecommendation.estimated_savings?.annual_kwh?.toFixed(1) || 0} kWh
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        CO₂ Reduction
                      </Typography>
                      <Typography variant="h6" color="success.main">
                        {selectedRecommendation.estimated_savings?.co2_reduction_kg?.toFixed(1) || 0} kg
                      </Typography>
                    </Box>
                  </Box>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Confidence Level: {(selectedRecommendation.confidence * 100).toFixed(0)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={selectedRecommendation.confidence * 100}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </AccordionDetails>
              </Accordion>

              {/* Implementation Steps */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Implementation Steps</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <List dense>
                    {selectedRecommendation.implementation_steps?.map((step: string, index: number) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={`${index + 1}. ${step}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>

              {/* Agent Source */}
              <Box sx={{ mt: 2, p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Generated by: <strong>{selectedRecommendation.agent_source}</strong>
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Created: {new Date(selectedRecommendation.created_at).toLocaleString()}
                </Typography>
              </Box>
            </DialogContent>

            <DialogActions>
              <Button onClick={handleCloseDialog}>
                Cancel
              </Button>
              <Button
                onClick={() => handleStatusUpdate(selectedRecommendation.id, 'dismissed')}
                color="error"
                startIcon={<CancelIcon />}
              >
                Dismiss
              </Button>
              <Button
                onClick={() => handleStatusUpdate(selectedRecommendation.id, 'implemented')}
                color="success"
                variant="contained"
                startIcon={<CheckCircleIcon />}
              >
                Mark as Implemented
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </>
  );
};

export default RecommendationsWidget;